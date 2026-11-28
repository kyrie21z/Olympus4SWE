"""Static architecture fitness tests enforcing module dependency boundaries.

These tests mechanically protect the layering documented in ``AGENTS.md`` and
``docs/ARCHITECTURE.md`` so unintended cross-layer coupling fails loudly in CI.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OLYMPUS_SRC = REPO_ROOT / "src" / "olympus"
WORKFLOWS_DIR = OLYMPUS_SRC / "graph" / "workflows"
SKILLS_DIR = OLYMPUS_SRC / "skills"

VENDOR_LLM_SDKS = frozenset({"openai", "anthropic"})
WEB_AND_CLI_FRAMEWORKS = frozenset({"fastapi", "uvicorn", "typer"})
OLYMPUS_SUBPACKAGES = frozenset(
    {
        "core",
        "providers",
        "tools",
        "harness",
        "stages",
        "graph",
        "session",
        "web",
        "cli",
    }
)


@dataclass(frozen=True)
class ImportEdge:
    """One resolved import target discovered via AST inspection."""

    file: Path
    lineno: int
    target: str

    @property
    def top_level(self) -> str:
        return self.target.split(".", 1)[0]

    @property
    def olympus_subpackage(self) -> str | None:
        parts = self.target.split(".")
        if len(parts) >= 2 and parts[0] == "olympus":
            return parts[1]
        return None


def _module_name_for_file(file_path: Path, src_root: Path) -> str:
    rel = file_path.relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative_module(
    current_module: str,
    *,
    is_package_init: bool,
    level: int,
    module: str | None,
) -> str:
    """Resolve an AST ``ImportFrom`` relative import to a dotted module path."""
    parts = current_module.split(".") if current_module else []
    base = parts if is_package_init else parts[:-1]
    ascend = level - 1
    if ascend > len(base):
        resolved_base: list[str] = []
    elif ascend > 0:
        resolved_base = base[:-ascend]
    else:
        resolved_base = list(base)
    if module:
        resolved_base.extend(module.split("."))
    return ".".join(resolved_base)


def collect_imports_from_source(
    source: str,
    *,
    file_path: Path,
    current_module: str,
    is_package_init: bool = False,
) -> list[ImportEdge]:
    """Parse Python source and return all resolved import targets."""
    tree = ast.parse(source, filename=str(file_path))
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(ImportEdge(file=file_path, lineno=node.lineno, target=alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                base_target = _resolve_relative_module(
                    current_module,
                    is_package_init=is_package_init,
                    level=node.level,
                    module=node.module,
                )
                if node.module:
                    edges.append(
                        ImportEdge(file=file_path, lineno=node.lineno, target=base_target)
                    )
                else:
                    for alias in node.names:
                        if alias.name == "*":
                            if base_target:
                                edges.append(
                                    ImportEdge(
                                        file=file_path, lineno=node.lineno, target=base_target
                                    )
                                )
                        else:
                            combined = (
                                f"{base_target}.{alias.name}" if base_target else alias.name
                            )
                            edges.append(
                                ImportEdge(file=file_path, lineno=node.lineno, target=combined)
                            )
            elif node.module:
                if node.module == "olympus":
                    for alias in node.names:
                        target = (
                            "olympus" if alias.name == "*" else f"olympus.{alias.name}"
                        )
                        edges.append(
                            ImportEdge(file=file_path, lineno=node.lineno, target=target)
                        )
                else:
                    edges.append(
                        ImportEdge(file=file_path, lineno=node.lineno, target=node.module)
                    )
    return edges


def _iter_package_imports(package_dir: Path) -> list[ImportEdge]:
    src_root = REPO_ROOT / "src"
    edges: list[ImportEdge] = []
    for py_file in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        edges.extend(
            collect_imports_from_source(
                source,
                file_path=py_file,
                current_module=_module_name_for_file(py_file, src_root),
                is_package_init=py_file.name == "__init__.py",
            )
        )
    return edges


def _format_violations(violations: list[ImportEdge], hint: str) -> str:
    lines = [
        f"  - {edge.file.relative_to(REPO_ROOT)}:{edge.lineno} imports '{edge.target}'"
        for edge in violations
    ]
    return "Architecture boundary violation(s):\n" + "\n".join(lines) + f"\nHint: {hint}"


def test_core_has_no_outer_layer_or_vendor_dependencies() -> None:
    """``olympus.core`` must stay a pure foundation with zero sibling or SDK imports."""
    forbidden_subpackages = OLYMPUS_SUBPACKAGES - {"core"}
    forbidden_roots = VENDOR_LLM_SDKS | WEB_AND_CLI_FRAMEWORKS | {"tests"}
    violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC / "core")
        if edge.top_level in forbidden_roots
        or edge.olympus_subpackage in forbidden_subpackages
    ]
    assert not violations, _format_violations(
        violations,
        "Keep olympus.core independent of outer packages, vendor SDKs, and web/CLI frameworks.",
    )


def test_vendor_llm_sdks_confined_to_providers() -> None:
    """Only ``olympus.providers`` may import vendor LLM SDKs directly."""
    violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC)
        if edge.top_level in VENDOR_LLM_SDKS
        and "providers" not in edge.file.relative_to(OLYMPUS_SRC).parts[:1]
    ]
    assert not violations, _format_violations(
        violations,
        "Route all LLM calls through olympus.providers abstractions instead of vendor SDKs.",
    )


def test_domain_and_runtime_layers_do_not_import_web_or_cli() -> None:
    """Core and runtime packages must remain decoupled from Web/CLI presentation layers."""
    isolated_packages = ("core", "providers", "tools", "harness", "stages", "graph", "session")
    violations: list[ImportEdge] = []
    for pkg in isolated_packages:
        for edge in _iter_package_imports(OLYMPUS_SRC / pkg):
            if edge.top_level in WEB_AND_CLI_FRAMEWORKS or edge.olympus_subpackage in {
                "web",
                "cli",
            }:
                violations.append(edge)
    assert not violations, _format_violations(
        violations,
        "Web and CLI layers should project runtime state, not be imported by core/runtime modules.",
    )


def test_web_does_not_import_cli() -> None:
    """``olympus.web`` must stay decoupled from CLI commands and Typer."""
    violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC / "web")
        if edge.top_level == "typer" or edge.olympus_subpackage == "cli"
    ]
    assert not violations, _format_violations(
        violations,
        "Share domain logic through core/session/graph rather than importing olympus.cli in web.",
    )


def test_providers_and_tools_respect_bottom_layer_boundaries() -> None:
    """Bottom-layer adapters must not reach upward into orchestration or UI layers."""
    provider_forbidden = OLYMPUS_SUBPACKAGES - {"providers"}
    provider_violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC / "providers")
        if edge.olympus_subpackage in provider_forbidden
    ]
    assert not provider_violations, _format_violations(
        provider_violations,
        "Keep olympus.providers self-contained as a pluggable LLM compatibility layer.",
    )

    tools_forbidden = {"harness", "stages", "graph", "session", "web", "cli"}
    tools_violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC / "tools")
        if edge.olympus_subpackage in tools_forbidden
    ]
    assert not tools_violations, _format_violations(
        tools_violations,
        "Keep olympus.tools below harness/stages orchestration layers.",
    )


def test_production_code_never_imports_tests_or_evals() -> None:
    """Enforce one-way dependency: ``tests/evals/ -> src/olympus/``, never the reverse."""
    violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC)
        if edge.top_level == "tests"
    ]
    assert not violations, _format_violations(
        violations,
        "Production code must stay independent of tests/evals; expose read-only contracts instead.",
    )


def test_declarative_workflows_contain_only_yaml() -> None:
    """Declarative workflow templates under ``graph/workflows/`` must stay data-only YAML."""
    non_yaml = sorted(
        path
        for path in WORKFLOWS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() not in {".yaml", ".yml"}
    )
    assert not non_yaml, (
        "Unexpected non-YAML file(s) in src/olympus/graph/workflows/: "
        + ", ".join(str(p.relative_to(REPO_ROOT)) for p in non_yaml)
    )


def test_relative_import_resolution_catches_escape() -> None:
    """Verify the AST helper catches both relative and nested cross-layer imports."""
    sample = """
from ..web import server
from .. import cli
import openai

def helper() -> None:
    from ...providers.dialects import openai_chat
"""
    edges = collect_imports_from_source(
        sample,
        file_path=OLYMPUS_SRC / "core" / "sub" / "demo.py",
        current_module="olympus.core.sub.demo",
    )
    targets = {edge.target for edge in edges}
    assert "olympus.core.web" in targets
    assert "olympus.core.cli" in targets
    assert "openai" in targets
    assert "olympus.providers.dialects" in targets

    pkg_edges = collect_imports_from_source(
        "from ..web import server\nfrom olympus import cli, providers\n",
        file_path=OLYMPUS_SRC / "core" / "contracts.py",
        current_module="olympus.core.contracts",
    )
    assert {edge.target for edge in pkg_edges} == {
        "olympus.web",
        "olympus.cli",
        "olympus.providers",
    }


def test_all_builtin_workflow_manifests_skills_and_nodes_are_valid() -> None:
    """Auto-discover every builtin workflow YAML and verify manifest, graph, and skill contracts."""
    from olympus.graph.loader import (
        _NODE_TYPES,
        build_graph,
        describe_workflow,
        list_builtin_workflows,
        load_workflow,
    )
    from olympus.web.schemas import WorkflowSummary

    yaml_files = sorted(WORKFLOWS_DIR.glob("*.yaml"))
    assert yaml_files, "expected builtin workflow YAML files under src/olympus/graph/workflows/"
    discovered_names = [path.stem for path in yaml_files]
    assert list_builtin_workflows() == discovered_names

    valid_autonomy = {"gated", "autonomous_hard_stops", "read_only"}
    valid_stop_after = {"goal", "design", "plan", "execute", "verify", "report"}
    registered_nodes = set(_NODE_TYPES)

    for name in discovered_names:
        definition = load_workflow(name)
        assert definition.name == name
        assert definition.family.strip(), f"workflow '{name}' must declare non-empty family"
        assert definition.autonomy in valid_autonomy, (
            f"workflow '{name}' has invalid autonomy: {definition.autonomy!r}"
        )
        assert definition.stop_after in valid_stop_after, (
            f"workflow '{name}' has invalid stop_after: {definition.stop_after!r}"
        )
        assert definition.applicable_scenarios, (
            f"workflow '{name}' must declare non-empty applicable_scenarios"
        )
        assert definition.anti_patterns, (
            f"workflow '{name}' must declare non-empty anti_patterns"
        )

        for node in definition.nodes:
            assert node.type in registered_nodes, (
                f"workflow '{name}' node '{node.name}' uses unregistered type '{node.type}'"
            )
            for ref in node.skill_refs:
                skill_path = (SKILLS_DIR / f"{ref}.md").resolve()
                assert skill_path.is_relative_to(SKILLS_DIR.resolve()), (
                    f"workflow '{name}' node '{node.name}' skill_ref '{ref}' escapes skills dir"
                )
                assert skill_path.is_file(), (
                    f"workflow '{name}' node '{node.name}' references missing skill '{ref}' "
                    f"(expected {skill_path.relative_to(REPO_ROOT)})"
                )
                assert skill_path.read_text(encoding="utf-8").strip(), (
                    f"skill file {skill_path.relative_to(REPO_ROOT)} is empty"
                )

        # Graph construction and public Web API summary projection must both succeed cleanly.
        graph = build_graph(definition)
        assert graph.describe()["nodes"], f"workflow '{name}' produced an empty graph"
        summary = WorkflowSummary.model_validate(describe_workflow(definition))
        assert summary.name == name
        assert summary.family == definition.family
        assert summary.autonomy == definition.autonomy
        assert summary.stop_after == definition.stop_after


def test_legacy_workflow_execution_hashes_are_pinned_and_manifest_independent() -> None:
    """Protect legacy SDLC resume compatibility by pinning executable graph fingerprints."""
    from olympus.graph.loader import load_workflow

    pinned_legacy_hashes = {
        "full_sdlc_tdd": (
            "sha256:291faa5f5f3fb4d33efe1c28a579e04c4bc3ae0e085f0bce5c8110bf06a3cd7d"
        ),
        "product_blueprint": (
            "sha256:26841aed6e876cf8b7773727cc96e867053412b50c6851710ee1598987b2fae7"
        ),
        "repo_fast_patch": (
            "sha256:87aa62e27d58fb5fdea94754dace076bffbac8e458228d25b02f9f03d0b424a5"
        ),
    }
    for name, expected_hash in pinned_legacy_hashes.items():
        definition = load_workflow(name)
        assert definition.execution_hash() == expected_hash
        # Descriptive manifest metadata must never perturb execution_hash().
        mutated = definition.model_copy(
            update={
                "family": "custom_family",
                "autonomy": "read_only",
                "applicable_scenarios": ["synthetic scenario"],
                "anti_patterns": ["synthetic anti-pattern"],
            }
        )
        assert mutated.execution_hash() == expected_hash


def test_pydantic_minimum_version_supports_exclude_if() -> None:
    """Require pydantic>=2.12 in pyproject.toml when src/olympus uses Field(exclude_if=)."""
    uses_exclude_if = any(
        "exclude_if=" in py_file.read_text(encoding="utf-8")
        for py_file in sorted(OLYMPUS_SRC.rglob("*.py"))
        if "__pycache__" not in py_file.parts
    )
    if not uses_exclude_if:
        return

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps: list[str] = list(pyproject.get("project", {}).get("dependencies", []))
    pydantic_specs = [
        dep for dep in deps if re.match(r"^pydantic(?:\[[^\]]+\])?(?:[<>=!~]|$)", dep)
    ]
    assert pydantic_specs, "pyproject.toml must declare a pydantic dependency"

    match = re.search(r">=\s*(\d+)\.(\d+)(?:\.(\d+))?", pydantic_specs[0])
    assert match is not None, f"expected >= minimum version bound on {pydantic_specs[0]!r}"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (2, 12), (
        f"src/olympus/ uses Pydantic 2.12+ Field(exclude_if=...), "
        f"but pyproject.toml declares {pydantic_specs[0]!r}"
    )
