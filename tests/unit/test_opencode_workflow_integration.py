"""Offline integration tests wiring ``OpenCodeDriver`` through all 5 ``pstack`` workflows.

Each test exercises the real ``OpenCodeDriver`` subprocess adapter resolved through
``DriverRegistry()`` + ``ExecutionDriversConfig(role_drivers=...)`` against a lightweight,
deterministic local fake ``opencode`` CLI executable (zero network, zero real LLM calls).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from olympus.core.contracts import (
    ArchitectSketch,
    ArenaGraftItem,
    ArenaRubric,
    ArenaRubricCriterion,
    ArenaVerdict,
    BudgetPolicy,
    CandidateShape,
    EvidenceBrief,
    EvidenceFact,
    HillclimbConfig,
    HillclimbReport,
    InterrogateFinding,
    InterrogateReport,
    RunMode,
)
from olympus.core.store import RunStore
from olympus.graph import WorkflowRuntime, build_graph, load_workflow, run_graph
from olympus.harness.driver import DriverRegistry, ExecutionDriversConfig
from olympus.harness.opencode import _reset_opencode_availability_cache
from olympus.harness.structured import StructuredTurnError
from olympus.providers.base import MockLLMProvider
from olympus.stages.craft import CraftPause, CraftStage, _probe_bwrap
from olympus.stages.hillclimb import snapshot_frozen_harness


@pytest.fixture(autouse=True)
def _ci_bwrap_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use real bwrap on Linux; fail loudly when CI requires OS isolation."""
    _reset_opencode_availability_cache()
    if _probe_bwrap() is None:
        if os.environ.get("OLYMPUS_REQUIRE_ISOLATION_TESTS") == "1":
            pytest.fail(
                "OLYMPUS_REQUIRE_ISOLATION_TESTS=1 is set, but bwrap isolation is unavailable"
            )
        monkeypatch.setenv("OLYMPUS_ALLOW_UNSANDBOXED_REPRO", "1")
    yield
    _reset_opencode_availability_cache()


_FAKE_OPENCODE_TEMPLATE = '''#!{python_bin}
"""Deterministic fake OpenCode CLI driven by a JSON plan file."""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

PLAN_PATH = Path({plan_path!r})
LOG_PATH = Path({log_path!r})
COUNTER_PATH = Path({counter_path!r})


def _emit(record: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\\n")
    sys.stdout.flush()


def _extract_model(argv: list[str]) -> str:
    for idx, token in enumerate(argv):
        if token in ("--model", "-m") and idx + 1 < len(argv):
            return argv[idx + 1]
    return ""


def _next_call_index(model: str) -> int:
    COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COUNTER_PATH, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        counts = json.loads(raw) if raw else {{}}
        idx = int(counts.get(model, 0))
        counts[model] = idx + 1
        handle.seek(0)
        handle.truncate()
        json.dump(counts, handle)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return idx


def _append_log(record: dict[str, object]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _await_rendezvous(rendezvous: dict[str, object]) -> None:
    dir_path = Path(str(rendezvous["dir"]))
    marker = str(rendezvous["marker"])
    expected = int(rendezvous["expected"])
    timeout_s = float(rendezvous.get("timeout_s", 5.0))
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{{marker}}.ready").write_text(str(os.getpid()), encoding="utf-8")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(list(dir_path.glob("*.ready"))) >= expected:
            return
        time.sleep(0.02)


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "run" and "--help" in argv:
        sys.stdout.write(
            "USAGE\\n  opencode run [flags]\\nFLAGS\\n"
            "  --format choice  Output format (choices: default, json)\\n"
        )
        return 0

    model = _extract_model(argv)
    config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", "{{}}"))
    call_idx = _next_call_index(model)
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    actions = plan.get(model, [])
    if not actions:
        sys.stderr.write(f"No scripted actions for model {{model!r}}\\n")
        return 2
    action = actions[call_idx] if call_idx < len(actions) else actions[-1]

    ts_start = time.time()
    rendezvous = action.get("rendezvous")
    if isinstance(rendezvous, dict):
        _await_rendezvous(rendezvous)

    cwd = Path(os.getcwd())
    for rel_path, content in action.get("writes", {{}}).items():
        target = cwd / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")

    session_id = f"ses_{{model.replace('/', '_')}}_{{call_idx}}"
    _emit({{"type": "step_start", "sessionID": session_id, "part": {{"type": "step-start"}}}})
    if action.get("writes"):
        first_path = next(iter(action["writes"]))
        _emit(
            {{
                "type": "tool_use",
                "sessionID": session_id,
                "part": {{
                    "id": f"call_{{call_idx}}",
                    "type": "tool",
                    "tool": "write",
                    "state": {{
                        "status": "completed",
                        "input": {{"path": first_path}},
                        "output": "ok",
                        "time": {{"start": 1000, "end": 1010}},
                    }},
                }},
            }}
        )
    _emit(
        {{
            "type": "text",
            "sessionID": session_id,
            "part": {{"type": "text", "text": str(action.get("text", "ok"))}},
        }}
    )
    input_tokens = int(action.get("input_tokens", 20))
    output_tokens = int(action.get("output_tokens", 10))
    _emit(
        {{
            "type": "step_finish",
            "sessionID": session_id,
            "part": {{
                "type": "step-finish",
                "reason": "stop",
                "tokens": {{
                    "input": input_tokens,
                    "output": output_tokens,
                    "reasoning": 0,
                    "cache": {{"read": 0, "write": 0}},
                }},
            }},
        }}
    )
    ts_end = time.time()

    agent_cfg = config.get("agent", {{}}).get("olympus_turn", {{}})
    _append_log(
        {{
            "model": model,
            "call_idx": call_idx,
            "cwd": str(cwd.resolve()),
            "xdg_data": os.environ.get("XDG_DATA_HOME"),
            "xdg_config": os.environ.get("XDG_CONFIG_HOME"),
            "xdg_state": os.environ.get("XDG_STATE_HOME"),
            "opencode_db": os.environ.get("OPENCODE_DB"),
            "tmp_dir": os.environ.get("TMPDIR"),
            "permissions": agent_cfg.get("permission", {{}}),
            "steps_cap": agent_cfg.get("steps"),
            "ts_start": ts_start,
            "ts_end": ts_end,
        }}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _install_fake_opencode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, list[dict[str, Any]]],
    *,
    tag: str = "oc",
) -> Path:
    """Create a deterministic fake ``opencode`` executable and point OLYMPUS_OPENCODE_BIN at it."""
    oc_dir = tmp_path / f"fake_{tag}"
    oc_dir.mkdir(parents=True, exist_ok=True)
    plan_path = oc_dir / "plan.json"
    log_path = oc_dir / "invocations.jsonl"
    counter_path = oc_dir / "counters.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    fake_bin = oc_dir / "opencode"
    fake_bin.write_text(
        _FAKE_OPENCODE_TEMPLATE.format(
            python_bin=sys.executable,
            plan_path=str(plan_path),
            log_path=str(log_path),
            counter_path=str(counter_path),
        ),
        encoding="utf-8",
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _reset_opencode_availability_cache()
    monkeypatch.setenv("OLYMPUS_OPENCODE_BIN", str(fake_bin))
    return log_path


def _read_invocations(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init_git_repo(repo: Path, files: dict[str, str]) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        target = repo / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "ci-test@olympus.local")
    _git(repo, "config", "user.name", "Olympus CI")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")


# ---------------------------------------------------------------------------
# 1. pstack_craft: full pipeline + repair loop + DiffGate + pre-turn budget
# ---------------------------------------------------------------------------


def test_opencode_pstack_craft_e2e_repair_diff_gate_and_budget_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "craft_repo"
    _init_git_repo(
        repo,
        {
            "app.py": "def double(n: int) -> int:\n    return n + 1\n",
            "test_app.py": (
                "import unittest\n"
                "import app\n\n"
                "class TestApp(unittest.TestCase):\n"
                "    def test_double(self) -> None:\n"
                "        self.assertEqual(app.double(4), 8)\n"
            ),
            "secrets.py": "TOKEN = 'do-not-touch'\n",
        },
    )

    evidence = EvidenceBrief(
        summary="app.double returns n + 1 instead of n * 2",
        facts=[EvidenceFact(path="app.py", line=2, observation="returns n + 1")],
        repro_command=f"{sys.executable} -m unittest test_app.py",
        candidate_files=["app.py", "test_app.py"],
    ).model_dump_json()
    sketch = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(
                name="direct_fix",
                data_flow="app.double(n) -> n * 2",
                tradeoffs="Minimal 1-line fix",
            ),
            CandidateShape(
                name="shift_fix",
                data_flow="app.double(n) -> n << 1",
                tradeoffs="Bitwise shift obscures arithmetic intent",
            ),
        ],
        chosen_shape="direct_fix",
        rationale="Direct multiplication is clearest and handles all int inputs.",
        state_owners={"double": "app.py"},
        write_scope=["app.py"],
        verification_plan=[f"{sys.executable} -m unittest test_app.py"],
    ).model_dump_json()
    finding_report = InterrogateReport(
        verdict="findings",
        findings=[
            InterrogateFinding(
                finding_id="F1",
                severity="important",
                file_path="app.py",
                claim="double(n) still returns n + 2 instead of n * 2",
                repro_command=f"{sys.executable} -m unittest test_app.py",
            )
        ],
        summary="Blocking arithmetic bug remains in app.double.",
    ).model_dump_json()
    clean_report = InterrogateReport(
        verdict="pass",
        findings=[],
        summary="app.double cleanly returns n * 2 with zero slop.",
    ).model_dump_json()

    log_path = _install_fake_opencode(
        tmp_path,
        monkeypatch,
        {
            "fake/explorer": [{"text": evidence, "input_tokens": 25, "output_tokens": 15}],
            "fake/architect": [{"text": sketch, "input_tokens": 25, "output_tokens": 15}],
            "fake/builder": [
                {
                    "text": "First attempt with wrong formula.",
                    "writes": {"app.py": "def double(n: int) -> int:\n    return n + 2\n"},
                    "input_tokens": 30,
                    "output_tokens": 10,
                },
                {
                    "text": "Fixed root cause with comment.",
                    "writes": {
                        "app.py": "# temporary note\ndef double(n: int) -> int:\n    return n * 2\n"
                    },
                    "input_tokens": 30,
                    "output_tokens": 10,
                },
            ],
            "fake/hygiene": [
                {"text": "No hygiene edits on visit 1.", "input_tokens": 10, "output_tokens": 5},
                {
                    "text": "Stripped temporary comment.",
                    "writes": {"app.py": "def double(n: int) -> int:\n    return n * 2\n"},
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            ],
            "fake/interrogator": [
                {"text": finding_report, "input_tokens": 20, "output_tokens": 10},
                {"text": clean_report, "input_tokens": 20, "output_tokens": 10},
            ],
        },
        tag="craft_e2e",
    )

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="OpenCode pstack_craft E2E",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    role_drivers = ExecutionDriversConfig(
        role_drivers={
            "explorer": "opencode@fake/explorer",
            "architect": "opencode@fake/architect",
            "builder": "opencode@fake/builder",
            "hygiene": "opencode@fake/hygiene",
            "interrogator": "opencode@fake/interrogator",
        }
    )
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        apply_to_workspace=True,
        execution_drivers=role_drivers,
        driver_registry=DriverRegistry(),
    )

    result = run_graph(
        build_graph(load_workflow("pstack_craft")),
        {"run_id": run_id, "raw_input": "Fix app.double to return n * 2", "_runtime": runtime},
    )

    assert result.get("_paused_node") is None
    assert result["surface_verification"]["passed"] is True
    assert result["applied_paths"] == ["app.py"]
    assert (repo / "app.py").read_text(encoding="utf-8") == (
        "def double(n: int) -> int:\n    return n * 2\n"
    )

    invocations = _read_invocations(log_path)
    assert [row["model"] for row in invocations] == [
        "fake/explorer",
        "fake/architect",
        "fake/builder",
        "fake/hygiene",
        "fake/interrogator",
        "fake/builder",
        "fake/hygiene",
        "fake/interrogator",
    ]
    # Explorer/architect/interrogator must receive strict read-only permissions; builder/hygiene
    # receive scoped edit/write rules allowing app.py and denying *.
    for row in invocations:
        perms = row["permissions"]
        if row["model"] in ("fake/explorer", "fake/architect", "fake/interrogator"):
            assert perms.get("*") == "deny"
            assert "edit" not in perms
            assert "write" not in perms
            assert "bash" not in perms
        else:
            assert perms["edit"]["*"] == "deny"
            assert perms["edit"]["app.py"] == "allow"

    # OpenCodeDriver sets records_usage_internally=True: verify exact ledger settlement
    # with zero double-counting across all 8 turns (sum = 250 tokens, 8 calls).
    ledger = store.get_run(run_id)["budget_ledger"]
    assert ledger.llm_calls_used == 8
    assert ledger.total_tokens == 250
    receipt_events = [ev for ev in store.list_events(run_id) if ev.event_type == "turn.receipt"]
    assert len(receipt_events) == 8

    # Sub-case B: DiffGate catches out-of-scope mutation performed by OpenCodeDriver.
    _install_fake_opencode(
        tmp_path,
        monkeypatch,
        {
            "fake/builder_oos": [
                {
                    "text": "Modified out-of-scope secrets.py",
                    "writes": {
                        "app.py": "def double(n: int) -> int:\n    return n * 2\n",
                        "secrets.py": "TOKEN = 'mutated'\n",
                    },
                }
            ]
        },
        tag="craft_diff_gate",
    )
    oos_run = store.create_run(
        title="OpenCode DiffGate enforcement",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
    )
    oos_stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_oos",
        provider=MockLLMProvider([]),
        execution_drivers=ExecutionDriversConfig(
            role_drivers={"builder": "opencode@fake/builder_oos"}
        ),
        driver_registry=DriverRegistry(),
    )
    with pytest.raises(CraftPause) as pause_info:
        oos_stage.build_direct(
            run_id=oos_run,
            raw_input="Fix app.double",
            state={"architect_sketch": json.loads(sketch)},
        )
    assert "secrets.py" in pause_info.value.reason or "scope" in pause_info.value.reason.lower()
    assert (repo / "secrets.py").read_text(encoding="utf-8") == "TOKEN = 'do-not-touch'\n"

    # Sub-case C: records_usage_internally=True pre-turn budget enforcement on parse repair retry.
    budget_log = _install_fake_opencode(
        tmp_path,
        monkeypatch,
        {
            "fake/architect_bad": [
                {"text": "not valid json #1", "input_tokens": 25, "output_tokens": 15},
                {"text": "not valid json #2", "input_tokens": 25, "output_tokens": 15},
                {"text": sketch, "input_tokens": 25, "output_tokens": 15},
            ]
        },
        tag="craft_budget",
    )
    capped_run = store.create_run(
        title="OpenCode budget cap on structured repair retry",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_craft",
        workspace_root=repo,
        budget_policy=BudgetPolicy(max_llm_calls=2, max_total_tokens=100),
    )
    capped_stage = CraftStage(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes_budget",
        provider=MockLLMProvider([]),
        execution_drivers=ExecutionDriversConfig(
            role_drivers={"architect": "opencode@fake/architect_bad"}
        ),
        driver_registry=DriverRegistry(),
    )
    with pytest.raises(StructuredTurnError, match="budget_exhausted"):
        capped_stage.architect_sketch(run_id=capped_run, raw_input="Fix app.py", state={})
    # Attempt 3 must NEVER spawn a third OpenCodeDriver subprocess.
    assert len(_read_invocations(budget_log)) == 2
    capped_ledger = store.get_run(capped_run)["budget_ledger"]
    assert capped_ledger.llm_calls_used == 2
    assert capped_ledger.total_tokens == 80


# ---------------------------------------------------------------------------
# 2. pstack_bugfix: require_repro + failing_test_first SHA-256 freeze -> green
# ---------------------------------------------------------------------------


def test_opencode_pstack_bugfix_red_test_freeze_to_green_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "bugfix_repo"
    _init_git_repo(
        repo,
        {
            "pricing.py": (
                "def apply_discount(price: int, percent: int) -> int:\n"
                "    return price + (price * percent // 100)\n"
            ),
            "test_pricing.py": (
                "import unittest\n"
                "import pricing\n\n"
                "class TestPricing(unittest.TestCase):\n"
                "    def test_discount_reduces_price(self) -> None:\n"
                "        self.assertEqual(pricing.apply_discount(100, 20), 80)\n"
            ),
        },
    )
    orig_test_hash = _sha256(repo / "test_pricing.py")

    evidence = EvidenceBrief(
        summary="pricing.apply_discount adds instead of subtracts discount amount",
        facts=[EvidenceFact(path="pricing.py", line=2, observation="uses + instead of -")],
        repro_command=f"{sys.executable} -m unittest test_pricing.py",
        candidate_files=["pricing.py", "test_pricing.py"],
    ).model_dump_json()
    sketch = ArchitectSketch(
        candidate_shapes=[
            CandidateShape(
                name="root_cause_fix",
                data_flow="apply_discount(price, percent) -> price - (price * percent // 100)",
                tradeoffs="Minimal direct sign correction",
            ),
            CandidateShape(
                name="float_factor",
                data_flow="apply_discount(price, percent) -> int(price * (100 - percent) / 100)",
                tradeoffs="Introduces float rounding divergence",
            ),
        ],
        chosen_shape="root_cause_fix",
        rationale="Exact integer subtraction fixes the sign bug at the root cause.",
        state_owners={"apply_discount": "pricing.py"},
        write_scope=["pricing.py", "test_pricing.py"],
        verification_plan=[f"{sys.executable} -m unittest test_pricing.py"],
    ).model_dump_json()
    clean_review = InterrogateReport(
        verdict="pass",
        findings=[],
        summary="Root-cause fix verified with frozen red test turned green.",
    ).model_dump_json()

    log_path = _install_fake_opencode(
        tmp_path,
        monkeypatch,
        {
            "fake/bf_explorer": [{"text": evidence}],
            "fake/bf_architect": [{"text": sketch}],
            "fake/bf_builder": [
                {
                    "text": "Attempt 1 still fails red test.",
                    "writes": {
                        "pricing.py": (
                            "def apply_discount(price: int, percent: int) -> int:\n"
                            "    return price + 1\n"
                        )
                    },
                },
                {
                    "text": "Attempt 2 fixes sign bug.",
                    "writes": {
                        "pricing.py": (
                            "# cleanup comment\n"
                            "def apply_discount(price: int, percent: int) -> int:\n"
                            "    return price - (price * percent // 100)\n"
                        )
                    },
                },
            ],
            "fake/bf_hygiene": [
                {
                    "text": "Removed redundant comment.",
                    "writes": {
                        "pricing.py": (
                            "def apply_discount(price: int, percent: int) -> int:\n"
                            "    return price - (price * percent // 100)\n"
                        )
                    },
                }
            ],
            "fake/bf_interrogator": [{"text": clean_review}],
        },
        tag="bugfix",
    )

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="OpenCode pstack_bugfix E2E",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_bugfix",
        workspace_root=repo,
    )
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        apply_to_workspace=True,
        execution_drivers=ExecutionDriversConfig(
            role_drivers={
                "explorer": "opencode@fake/bf_explorer",
                "architect": "opencode@fake/bf_architect",
                "builder": "opencode@fake/bf_builder",
                "hygiene": "opencode@fake/bf_hygiene",
                "interrogator": "opencode@fake/bf_interrogator",
            }
        ),
        driver_registry=DriverRegistry(),
    )

    result = run_graph(
        build_graph(load_workflow("pstack_bugfix")),
        {"run_id": run_id, "raw_input": "Fix pricing discount sign bug", "_runtime": runtime},
    )

    assert result.get("_paused_node") is None
    assert result["evidence_brief"]["repro_verified"] is True
    assert result["failing_test_receipt"]["same_surface"] is True
    assert result["failing_test_receipt"]["red"]["verified"] is True
    assert result["failing_test_receipt"]["red"]["exit_code"] != 0
    assert result["failing_test_receipt"]["green"]["verified"] is True
    assert result["failing_test_receipt"]["green"]["exit_code"] == 0
    assert result["surface_verification"]["passed"] is True
    assert result["applied_paths"] == ["pricing.py"]
    assert _sha256(repo / "test_pricing.py") == orig_test_hash

    events = [ev.event_type for ev in store.list_events(run_id)]
    assert "craft.red_test_recorded" in events
    assert "craft.failing_test_still_red" in events
    assert "craft.failing_test_verified" in events

    models_called = [row["model"] for row in _read_invocations(log_path)]
    assert models_called == [
        "fake/bf_explorer",
        "fake/bf_architect",
        "fake/bf_builder",
        "fake/bf_builder",
        "fake/bf_hygiene",
        "fake/bf_interrogator",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "test_pricing.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 3. investigate: strict read-only permissions + zero host workspace mutation
# ---------------------------------------------------------------------------


def test_opencode_investigate_read_only_permissions_and_zero_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "investigate_repo"
    _init_git_repo(
        repo,
        {
            "pricing.py": (
                "def apply_discount(price: int, percent: int) -> int:\n"
                "    return price + (price * percent // 100)\n"
            ),
            "test_pricing.py": (
                "import unittest\n"
                "import pricing\n\n"
                "class TestPricing(unittest.TestCase):\n"
                "    def test_discount(self) -> None:\n"
                "        self.assertEqual(pricing.apply_discount(100, 20), 80)\n"
            ),
        },
    )
    before_head = _git(repo, "rev-parse", "HEAD")
    before_files = sorted(
        str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts
    )

    forensics_brief = EvidenceBrief(
        summary="pricing.apply_discount at pricing.py:2 adds discount instead of subtracting it.",
        facts=[
            EvidenceFact(
                path="pricing.py",
                line=2,
                observation="returns price + (price * percent // 100)",
            )
        ],
        repro_command=f"{sys.executable} -m unittest test_pricing.py",
        candidate_files=["pricing.py", "test_pricing.py"],
        unresolved_questions=["Should percent > 100 be clamped or raise ValueError?"],
    ).model_dump_json()

    log_path = _install_fake_opencode(
        tmp_path,
        monkeypatch,
        {"fake/inv_explorer": [{"text": forensics_brief, "input_tokens": 18, "output_tokens": 12}]},
        tag="investigate",
    )

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="OpenCode investigate read-only",
        mode=RunMode.EXECUTION,
        workflow_name="investigate",
        workspace_root=repo,
    )
    artifacts_root = tmp_path / "artifacts"
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=artifacts_root,
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        apply_to_workspace=True,  # Must still never mutate workspace in investigate.
        execution_drivers=ExecutionDriversConfig(
            role_drivers={"explorer": "opencode@fake/inv_explorer"}
        ),
        driver_registry=DriverRegistry(),
    )

    result = run_graph(
        build_graph(load_workflow("investigate")),
        {"run_id": run_id, "raw_input": "Why does apply_discount overcharge?", "_runtime": runtime},
    )

    assert result.get("_paused_node") is None
    assert result["route"] == "report"
    assert result["read_only"] is True
    assert "# Investigation Report" in result["investigation_report"]
    assert "`pricing.py:2`" in result["investigation_report"]

    invocations = _read_invocations(log_path)
    assert len(invocations) == 1
    inv = invocations[0]
    assert inv["permissions"] == {
        "*": "deny",
        "read": "allow",
        "grep": "allow",
        "glob": "allow",
        "list": "allow",
        "external_directory": "deny",
    }
    # Temp dir must stay outside the host repo so read-only turns never drop artifacts in repo.
    assert inv["tmp_dir"] is not None
    assert not Path(inv["tmp_dir"]).resolve().is_relative_to(repo.resolve())

    # Zero host workspace mutation invariant.
    assert _git(repo, "status", "--porcelain") == ""
    assert _git(repo, "rev-parse", "HEAD") == before_head
    after_files = sorted(
        str(p.relative_to(repo)) for p in repo.rglob("*") if ".git" not in p.parts
    )
    assert after_files == before_files
    assert not (tmp_path / "sandboxes").exists()


# ---------------------------------------------------------------------------
# 4. pstack_arena: concurrent OpenCode worktrees + XDG isolation + judge + graft
# ---------------------------------------------------------------------------


def test_opencode_pstack_arena_concurrent_worktrees_xdg_isolation_and_graft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "arena_repo"
    _init_git_repo(
        repo,
        {
            "calc.py": "def compute(x: int, y: int) -> int:\n    return 0\n",
            "test_calc.py": (
                "import unittest\n"
                "import calc\n\n"
                "class TestCalc(unittest.TestCase):\n"
                "    def test_compute(self) -> None:\n"
                "        self.assertEqual(calc.compute(2, 3), 5)\n"
            ),
        },
    )

    evidence = EvidenceBrief(
        summary="calc.compute currently returns 0 instead of summing x and y",
        facts=[EvidenceFact(path="calc.py", line=2, observation="returns literal 0")],
        repro_command=f"{sys.executable} -m unittest test_calc.py",
        candidate_files=["calc.py", "test_calc.py"],
    ).model_dump_json()
    rubric = ArenaRubric(
        summary="Compare candidate implementations of calc.compute",
        write_scope=["calc.py"],
        verification_plan=[f"{sys.executable} -m unittest test_calc.py"],
        criteria=[
            ArenaRubricCriterion(
                criterion_id="correctness", weight=0.5, description="Passes unit tests"
            ),
            ArenaRubricCriterion(
                criterion_id="simplicity", weight=0.3, description="Minimal clean code"
            ),
            ArenaRubricCriterion(
                criterion_id="clarity", weight=0.2, description="Type-safe coercion"
            ),
        ],
    ).model_dump_json()
    verdict = ArenaVerdict(
        winner_id="cand_1",
        scores={
            "cand_1": {"correctness": 9.5, "simplicity": 9.0, "clarity": 8.0},
            "cand_2": {"correctness": 9.5, "simplicity": 8.0, "clarity": 9.5},
        },
        graft_items=[
            ArenaGraftItem(
                source_candidate_id="cand_2",
                idea="Coerce inputs with int() as in cand_2.",
                target_files=["calc.py"],
            )
        ],
        convergence_mode="winner_plus_graft",
        rationale="cand_1 is cleanest; graft cand_2's int() coercion.",
    ).model_dump_json()

    rendezvous_dir = tmp_path / "arena_rendezvous"
    log_path = _install_fake_opencode(
        tmp_path,
        monkeypatch,
        {
            "fake/arena_explorer": [{"text": evidence}],
            "fake/arena_architect": [{"text": rubric}],
            "fake/arena_cand1": [
                {
                    "text": "cand_1 implemented simple sum",
                    "rendezvous": {"dir": str(rendezvous_dir), "marker": "cand1", "expected": 2},
                    "writes": {
                        "calc.py": (
                            "def compute(x: int, y: int) -> int:\n"
                            "    # slop comment\n"
                            "    return x + y\n"
                        )
                    },
                }
            ],
            "fake/arena_cand2": [
                {
                    "text": "cand_2 implemented coerced sum",
                    "rendezvous": {"dir": str(rendezvous_dir), "marker": "cand2", "expected": 2},
                    "writes": {
                        "calc.py": (
                            "def compute(x: int, y: int) -> int:\n"
                            "    return int(x) + int(y)\n"
                        )
                    },
                }
            ],
            "fake/arena_judge": [{"text": verdict}],
            "fake/arena_graft": [
                {
                    "text": "grafted int() coercion onto winner",
                    "writes": {
                        "calc.py": (
                            "def compute(x: int, y: int) -> int:\n"
                            "    # slop comment\n"
                            "    return int(x) + int(y)\n"
                        )
                    },
                }
            ],
            "fake/arena_hygiene": [
                {
                    "text": "stripped slop comment",
                    "writes": {
                        "calc.py": (
                            "def compute(x: int, y: int) -> int:\n"
                            "    return int(x) + int(y)\n"
                        )
                    },
                }
            ],
        },
        tag="arena",
    )

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="OpenCode pstack_arena E2E",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_arena",
        workspace_root=repo,
    )
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        apply_to_workspace=True,
        execution_drivers=ExecutionDriversConfig(
            role_drivers={
                "explorer": "opencode@fake/arena_explorer",
                "architect": "opencode@fake/arena_architect",
                "arena_runner_1": "opencode@fake/arena_cand1",
                "arena_runner_2": "opencode@fake/arena_cand2",
                "arena_judge": "opencode@fake/arena_judge",
                "builder": "opencode@fake/arena_graft",
                "hygiene": "opencode@fake/arena_hygiene",
            }
        ),
        driver_registry=DriverRegistry(),
    )

    final_state = run_graph(
        build_graph(load_workflow("pstack_arena")),
        {
            "run_id": run_id,
            "raw_input": "Fix calc.compute so it returns the integer sum of x and y.",
            "_runtime": runtime,
        },
    )

    assert final_state["route"] == "pass"
    assert final_state["arena_verdict"]["winner_id"] == "cand_1"
    assert final_state["surface_verification"]["passed"] is True
    assert final_state["applied_paths"] == ["calc.py"]
    assert (repo / "calc.py").read_text(encoding="utf-8") == (
        "def compute(x: int, y: int) -> int:\n    return int(x) + int(y)\n"
    )

    invocations = _read_invocations(log_path)
    by_model = {row["model"]: row for row in invocations}
    assert set(by_model) == {
        "fake/arena_explorer",
        "fake/arena_architect",
        "fake/arena_cand1",
        "fake/arena_cand2",
        "fake/arena_judge",
        "fake/arena_graft",
        "fake/arena_hygiene",
    }

    cand1 = by_model["fake/arena_cand1"]
    cand2 = by_model["fake/arena_cand2"]
    # Concurrent candidates must run in distinct git worktrees with distinct ephemeral XDG dirs.
    assert cand1["cwd"] != cand2["cwd"]
    assert cand1["cwd"] != str(repo.resolve())
    assert cand2["cwd"] != str(repo.resolve())
    assert cand1["xdg_data"] != cand2["xdg_data"]
    assert cand1["xdg_config"] != cand2["xdg_config"]
    assert cand1["xdg_state"] != cand2["xdg_state"]
    assert cand1["opencode_db"] is None
    assert cand2["opencode_db"] is None
    # Both candidate rendezvous markers prove concurrent overlap occurred.
    assert (rendezvous_dir / "cand1.ready").is_file()
    assert (rendezvous_dir / "cand2.ready").is_file()
    # Ephemeral XDG state dirs must be cleaned up after turn completion.
    assert not Path(cand1["xdg_data"]).exists()
    assert not Path(cand2["xdg_data"]).exists()
    # Independent arena_judge must run read-only.
    assert by_model["fake/arena_judge"]["permissions"].get("*") == "deny"
    assert "edit" not in by_model["fake/arena_judge"]["permissions"]

    # Clean worktree teardown: candidate worktrees must be removed (only main + craft remain,
    # and no arena candidate worktree leaks).
    wt_list = _git(repo, "worktree", "list", "--porcelain")
    assert "arena-cand" not in wt_list


# ---------------------------------------------------------------------------
# 5. pstack_hillclimb: DiffGate + frozen harness anti-tamper + keep/revert
# ---------------------------------------------------------------------------


def test_opencode_pstack_hillclimb_diff_gate_anti_tamper_and_keep_revert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "hillclimb_repo"
    _init_git_repo(
        repo,
        {
            "solver.py": (
                "def score_value() -> int:\n"
                "    return 100\n\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    for i in range(len(nums)):\n"
                "        for j in range(i + 1, len(nums)):\n"
                "            if nums[i] == nums[j]:\n"
                "                return True\n"
                "    return False\n"
            ),
            "bench.py": (
                "import solver\n"
                "assert solver.has_duplicate([1, 2, 1]) is True\n"
                "assert solver.has_duplicate([1, 2, 3]) is False\n"
                "print(f'SCORE={solver.score_value()}')\n"
            ),
            "test_solver.py": (
                "import unittest\n"
                "import solver\n\n"
                "class TestSolver(unittest.TestCase):\n"
                "    def test_dup(self) -> None:\n"
                "        self.assertTrue(solver.has_duplicate([1, 2, 1]))\n"
                "        self.assertFalse(solver.has_duplicate([1, 2, 3]))\n"
            ),
            "notes.txt": "keep untouched\n",
        },
    )
    orig_bench_hash = _sha256(repo / "bench.py")

    evidence = EvidenceBrief(
        summary="solver.score_value returns 100; bench.py reports SCORE=100",
        facts=[EvidenceFact(path="solver.py", line=2, observation="returns 100")],
        repro_command=f"{sys.executable} bench.py",
        candidate_files=["solver.py", "bench.py"],
    ).model_dump_json()

    log_path = _install_fake_opencode(
        tmp_path,
        monkeypatch,
        {
            "fake/hc_explorer": [{"text": evidence, "input_tokens": 20, "output_tokens": 10}],
            "fake/hc_builder": [
                # Iteration 1: out-of-scope file mutation -> DiffGate rejects & reverts.
                {
                    "text": "HYPOTHESIS: Touch out-of-scope notes.txt alongside solver.py",
                    "writes": {
                        "solver.py": (
                            "def score_value() -> int:\n    return 50\n"
                            "def has_duplicate(nums: list[int]) -> bool:\n"
                            "    return len(set(nums)) < len(nums)\n"
                        ),
                        "notes.txt": "out-of-scope edit\n",
                    },
                    "input_tokens": 20,
                    "output_tokens": 10,
                },
                # Iteration 2: frozen harness tamper -> rejected_tamper & reverted.
                {
                    "text": "HYPOTHESIS: Tamper with frozen bench.py",
                    "writes": {"bench.py": "print('SCORE=0')\n"},
                    "input_tokens": 20,
                    "output_tokens": 10,
                },
                # Iteration 3: regression (100 -> 150) -> reverted cleanly.
                {
                    "text": "HYPOTHESIS: Regress score to 150",
                    "writes": {
                        "solver.py": (
                            "def score_value() -> int:\n    return 150\n"
                            "def has_duplicate(nums: list[int]) -> bool:\n"
                            "    return len(set(nums)) < len(nums)\n"
                        )
                    },
                    "input_tokens": 20,
                    "output_tokens": 10,
                },
                # Iteration 4: genuine improvement (100 -> 12, below target_value=15) -> kept!
                {
                    "text": "HYPOTHESIS: Use hash set and cut score to 12",
                    "writes": {
                        "solver.py": (
                            "def score_value() -> int:\n    return 12\n"
                            "def has_duplicate(nums: list[int]) -> bool:\n"
                            "    return len(set(nums)) < len(nums)\n"
                        )
                    },
                    "input_tokens": 20,
                    "output_tokens": 10,
                },
            ],
        },
        tag="hillclimb",
    )

    store = RunStore(tmp_path / "olympus.sqlite")
    run_id = store.create_run(
        title="OpenCode pstack_hillclimb E2E",
        mode=RunMode.EXECUTION,
        workflow_name="pstack_hillclimb",
        workspace_root=repo,
    )
    runtime = WorkflowRuntime(
        store=store,
        workspace_root=repo,
        artifacts_root=tmp_path / "artifacts",
        sandbox_root=tmp_path / "sandboxes",
        provider=MockLLMProvider([]),
        apply_to_workspace=True,
        execution_drivers=ExecutionDriversConfig(
            role_drivers={
                "explorer": "opencode@fake/hc_explorer",
                "builder": "opencode@fake/hc_builder",
            }
        ),
        driver_registry=DriverRegistry(),
    )
    hc_config = HillclimbConfig(
        frozen_harness_paths=["bench.py"],
        metric_command=f"{sys.executable} bench.py",
        metric_regex=r"SCORE=([0-9]+(?:\.[0-9]+)?)",
        direction="minimize",
        target_value=15.0,
        max_iterations=5,
        plateau_patience=4,
        write_scope=["solver.py"],
    )

    result = run_graph(
        build_graph(load_workflow("pstack_hillclimb")),
        {
            "run_id": run_id,
            "raw_input": "Minimize SCORE in bench.py below 15 by optimizing solver.py",
            "hillclimb_config": hc_config.model_dump(mode="json"),
            "verification_commands": [f"{sys.executable} -m unittest test_solver.py"],
            "_runtime": runtime,
        },
    )

    assert result.get("_paused_node") is None
    report = HillclimbReport.model_validate(result["hillclimb_report"])
    assert report.baseline_metric == 100.0
    assert report.best_metric == 12.0
    assert report.kept_iterations == 1
    assert report.reverted_iterations == 3
    assert report.stopped_reason == "target_reached"
    assert [row.decision for row in report.rows] == [
        "error",
        "rejected_tamper",
        "revert",
        "keep",
    ]
    assert "scope" in report.rows[0].reason.lower() or "notes.txt" in report.rows[0].reason

    assert result["surface_verification"]["passed"] is True
    assert result["applied_paths"] == ["solver.py"]
    assert _sha256(repo / "bench.py") == orig_bench_hash
    assert snapshot_frozen_harness(repo, ["bench.py"]) == {"bench.py": orig_bench_hash}
    assert (repo / "notes.txt").read_text(encoding="utf-8") == "keep untouched\n"
    assert "return 12" in (repo / "solver.py").read_text(encoding="utf-8")

    invocations = _read_invocations(log_path)
    assert [row["model"] for row in invocations] == [
        "fake/hc_explorer",
        "fake/hc_builder",
        "fake/hc_builder",
        "fake/hc_builder",
        "fake/hc_builder",
    ]
    # Verify OpenCode declarative permissions allow solver.py and deny frozen bench.py & *.
    builder_perms = invocations[1]["permissions"]
    assert builder_perms["edit"]["*"] == "deny"
    assert builder_perms["edit"]["solver.py"] == "allow"
    assert builder_perms["edit"]["bench.py"] == "deny"

    # Verify internal usage settlement (5 turns x 30 tokens = 150 tokens, zero double-count).
    ledger = store.get_run(run_id)["budget_ledger"]
    assert ledger.llm_calls_used == 5
    assert ledger.total_tokens == 150
