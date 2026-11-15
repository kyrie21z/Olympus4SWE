"""Tests for the Docker-free SWE-Interact benchmark adapter and E2E pipeline."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from olympus.core.contracts import AmbiguityItem, CheckKind
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolCall, ToolSpec
from tests.evals.benchmarks.swe_interact import (
    PersonaUserOracle,
    apply_hidden_test_patch,
    extract_hidden_slots,
    extract_patch_target_paths,
    extract_persona_behavior,
    load_swe_interact_task_dir,
    parse_dockerfile,
    parse_task_toml,
    run_swe_interact_e2e,
)
from tests.evals.schemas import LeakGuard

_SAMPLE_TOML = """
schema_version = "1.3"

[task]
name = "scale-ai/deepswe_sample-converter"

[metadata]
task_id = "sample-converter"
display_title = "Add converter helpers"
display_description = "Add helpers to normalize records cleanly."
category = "feature_request"
language = "python"
repository_url = "https://example.com/repo.git"
base_commit_hash = "0123456789abcdef0123456789abcdef01234567"
"""

_SAMPLE_DOCKERFILE = """
FROM public.ecr.aws/x8v8d7g8/mars-base:latest
WORKDIR /app

RUN git clone https://example.com/repo.git . && \\
    git checkout 0123456789abcdef0123456789abcdef01234567

RUN poetry config virtualenvs.create false && \\
    poetry install --with dev --no-interaction

RUN pip install pytest-cov tabulate -e .

CMD ["/bin/bash"]
"""

_SAMPLE_INSTRUCTION = """Normalize records cleanly.

- `normalize_record(data, strict=False)` lives in `sample_pkg.convert`.
- Empty dictionary inputs `{}` return `{}` rather than raising `ValueError`.
"""

_SAMPLE_PERSONA = f"""# Shared invariants
Keep replies concise.

# Disclosure pattern: vibecoder
Answer one question directly without dumping everything.

{_SAMPLE_INSTRUCTION}
"""


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def test_parse_task_toml_dockerfile_and_persona_helpers() -> None:
    meta = parse_task_toml(_SAMPLE_TOML)
    assert meta["task_id"] == "sample-converter"
    assert meta["display_title"] == "Add converter helpers"
    assert meta["language"] == "python"

    repo_url, base_sha, pip_pkgs, uses_poetry = parse_dockerfile(_SAMPLE_DOCKERFILE)
    assert repo_url == "https://example.com/repo.git"
    assert base_sha == "0123456789abcdef0123456789abcdef01234567"
    assert pip_pkgs == ["pytest-cov", "tabulate"]
    assert uses_poetry is True

    behavior = extract_persona_behavior(_SAMPLE_PERSONA, _SAMPLE_INSTRUCTION)
    assert "Keep replies concise." in behavior
    assert "normalize_record(data, strict=False)" not in behavior

    slots = extract_hidden_slots(_SAMPLE_INSTRUCTION, meta["display_description"])
    assert len(slots) == 2
    assert "sample_pkg.convert" in slots[0].forbidden_leak_terms


def test_persona_user_oracle_rule_fallback_and_leak_redaction(tmp_path: Path) -> None:
    task_dir = tmp_path / "sample-task"
    (task_dir / "environment" / "user-server").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text(_SAMPLE_TOML, encoding="utf-8")
    (task_dir / "instruction.md").write_text(_SAMPLE_INSTRUCTION, encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text(_SAMPLE_DOCKERFILE, encoding="utf-8")
    (task_dir / "environment" / "user-server" / "persona.md").write_text(
        _SAMPLE_PERSONA,
        encoding="utf-8",
    )
    (task_dir / "tests" / "test.patch").write_text(
        "diff --git a/test.sh b/test.sh\n",
        encoding="utf-8",
    )
    task = load_swe_interact_task_dir(task_dir)

    rule_oracle = PersonaUserOracle(task)
    replies = rule_oracle.reply(
        [
            AmbiguityItem(
                id="Q1",
                question="What should Empty dictionary inputs return?",
                impact="Empty input return behavior",
            )
        ]
    )
    assert len(replies) == 1
    assert replies[0].revealed is True
    assert "{}" in replies[0].answer

    # LLM mode: unsolicited mention of S1 symbol when matching S2 should trigger LeakGuard
    def leaky_responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages, tools
        return LLMResponse(
            content=json.dumps(
                {
                    "answers": [
                        {
                            "question_index": 0,
                            "answer": "Return {} for empty dict; put helper in sample_pkg.convert.",
                            "matched_slot_ids": ["S2"],
                        }
                    ]
                }
            )
        )

    llm_oracle = PersonaUserOracle(task, provider=MockLLMProvider(responder=leaky_responder))
    llm_replies = llm_oracle.reply(
        [
            AmbiguityItem(
                id="Q1",
                question="Empty dictionary inputs behavior?",
                impact="Empty input handling",
            )
        ]
    )
    assert llm_oracle.leakage_detected is True
    assert "sample_pkg.convert" in llm_oracle.leak_terms
    assert "[redacted]" in llm_replies[0].answer
    assert LeakGuard.redact_terms("to_inline_table uses table", ["table"]) == (
        "to_inline_table uses [redacted]"
    )


def test_apply_hidden_test_patch_resets_conflicting_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "dev@example.com")
    _git(repo, "config", "user.name", "Dev")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_base.py").write_text(
        "def test_ok():\n    assert 1 == 1\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    # Model accidentally edits tests/test_base.py and creates test.sh
    (repo / "tests" / "test_base.py").write_text("CORRUPTED\n", encoding="utf-8")
    (repo / "test.sh").write_text("#!/bin/bash\nexit 99\n", encoding="utf-8")

    test_patch = (
        "diff --git a/test.sh b/test.sh\n"
        "new file mode 100755\n"
        "index 0000000..e69de29\n"
        "--- /dev/null\n"
        "+++ b/test.sh\n"
        "@@ -0,0 +1,2 @@\n"
        "+#!/bin/bash\n"
        "+echo ok-$1\n"
        "diff --git a/tests/test_base.py b/tests/test_base.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/tests/test_base.py\n"
        "+++ b/tests/test_base.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def test_ok():\n"
        "     assert 1 == 1\n"
        "+    assert 2 == 2\n"
    )
    assert extract_patch_target_paths(test_patch) == ["test.sh", "tests/test_base.py"]

    test_sh = apply_hidden_test_patch(repo, test_patch)
    assert test_sh.is_file()
    assert "assert 2 == 2" in (repo / "tests" / "test_base.py").read_text(encoding="utf-8")
    assert "echo ok-$1" in test_sh.read_text(encoding="utf-8")


def test_swe_interact_e2e_local_repo_real_pytest(tmp_path: Path) -> None:
    # 1. Seed a real upstream git repo
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init")
    _git(upstream, "config", "user.email", "dev@example.com")
    _git(upstream, "config", "user.name", "Dev")
    (upstream / "sample_pkg").mkdir()
    (upstream / "sample_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (upstream / "sample_pkg" / "core.py").write_text(
        "def identity(x):\n    return x\n",
        encoding="utf-8",
    )
    (upstream / "tests").mkdir()
    (upstream / "tests" / "test_base.py").write_text(
        "from sample_pkg.core import identity\n\n"
        "def test_identity():\n"
        "    assert identity(42) == 42\n",
        encoding="utf-8",
    )
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-m", "initial")
    base_sha = _git(upstream, "rev-parse", "HEAD")

    # 2. Generate a valid unified diff for hidden test.patch (test.sh + tests/test_new.py)
    probe = tmp_path / "probe"
    subprocess.run(["git", "clone", str(upstream), str(probe)], check=True, capture_output=True)
    (probe / "test.sh").write_text(
        "#!/bin/bash\n"
        "set -e\n"
        'if [ "$1" = "base" ]; then\n'
        "  python -m pytest tests/test_base.py\n"
        "else\n"
        "  python -m pytest tests/test_new.py\n"
        "fi\n",
        encoding="utf-8",
    )
    (probe / "tests" / "test_new.py").write_text(
        "from sample_pkg.core import normalize\n\n"
        "def test_normalize_empty():\n"
        "    assert normalize({}) == {'empty': True}\n\n"
        "def test_normalize_nonempty():\n"
        "    assert normalize({'a': 1}) == {'a': 1}\n",
        encoding="utf-8",
    )
    _git(probe, "add", "test.sh", "tests/test_new.py")
    hidden_test_patch = _git(probe, "diff", "--cached", "--binary")

    # 3. Materialize SWE-Interact task fixture
    task_dir = tmp_path / "task"
    (task_dir / "environment" / "user-server").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        f"""
[metadata]
task_id = "mini-normalize"
display_title = "Add normalize helper"
display_description = "Add normalize(data) to sample_pkg.core."
language = "python"
repository_url = "{upstream}"
base_commit_hash = "{base_sha}"
""",
        encoding="utf-8",
    )
    instr = (
        "Add `normalize(data)` to `sample_pkg.core`.\n\n"
        "- For empty dict inputs `{}` it must return `{'empty': True}`.\n"
        "- For non-empty dicts return a shallow copy.\n"
    )
    (task_dir / "instruction.md").write_text(instr, encoding="utf-8")
    (task_dir / "environment" / "user-server" / "persona.md").write_text(
        f"# Style\nConcise.\n\n{instr}",
        encoding="utf-8",
    )
    (task_dir / "tests" / "test.patch").write_text(hidden_test_patch + "\n", encoding="utf-8")
    task = load_swe_interact_task_dir(task_dir, task_name="mini-normalize")

    seen_worker_prompt: list[str] = []

    def agent_responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        sys_msg = messages[0].content
        user_msg = messages[1].content
        assert "test_new.py" not in sys_msg and "test_new.py" not in user_msg
        if "Requirements Alignment Architect" in sys_msg:
            return LLMResponse(
                content=json.dumps(
                    {
                        "spec_id": "spec-mini",
                        "version": 1,
                        "goal": "Implement normalize(data) in sample_pkg/core.py",
                        "requirements": [
                            {
                                "id": "R1",
                                "description": "Add normalize(data) to sample_pkg.core",
                                "acceptance_criteria": ["Empty dict and non-empty dict handled"],
                            }
                        ],
                        "open_ambiguities": [
                            {
                                "id": "AMB-1",
                                "question": "What should normalize({}) return when empty?",
                                "impact": "Empty dict behavior",
                                "blocking": True,
                            }
                        ],
                    }
                )
            )
        if "Task Planner" in sys_msg:
            m = re.search(r"Set spec_id='([^']+)' and spec_hash='([^']+)'", user_msg)
            assert m is not None
            return LLMResponse(
                content=json.dumps(
                    {
                        "plan_id": "plan-mini",
                        "spec_id": m.group(1),
                        "spec_hash": m.group(2),
                        "tasks": [
                            {
                                "task_id": "T1",
                                "title": "Implement normalize",
                                "objective": "Implement normalize in sample_pkg/core.py",
                                "requirement_refs": ["R1"],
                                "write_scope": ["sample_pkg/core.py"],
                                "acceptance_checks": [
                                    {
                                        "id": "C1",
                                        "kind": CheckKind.COMMAND.value,
                                        "description": "Verify normalize({}) return value",
                                        "command": (
                                            'python3 -c "from sample_pkg.core import normalize; '
                                            "assert normalize({}) == {'empty': True}\""
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
        if "Execution Engineer" in sys_msg:
            if not any(msg.role == "tool" for msg in messages):
                seen_worker_prompt.append(user_msg)
                ws_match = re.search(r"Sandbox workspace root: (.+)", user_msg)
                assert ws_match is not None
                ws = Path(ws_match.group(1).strip())
                # Worker derives exact return value from clarified oracle answer in prompt!
                code = (
                    "def identity(x):\n    return x\n\n"
                    "def normalize(data):\n"
                    "    if not data:\n"
                    "        return {'empty': True}\n"
                    "    return dict(data)\n"
                )
                return LLMResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="w1",
                            name="write_file",
                            arguments={
                                "path": str(ws / "sample_pkg" / "core.py"),
                                "content": code,
                            },
                        ),
                    ),
                )
            return LLMResponse(content="Implemented normalize(data).")
        if "Independent Verifier" in sys_msg:
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1",
                        "attempt": 1,
                        "passed": True,
                        "summary": "Looks great.",
                        "findings": [],
                    }
                )
            )
        raise AssertionError(f"Unexpected prompt: {sys_msg}")

    def oracle_responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages, tools
        return LLMResponse(
            content=json.dumps(
                {
                    "answers": [
                        {
                            "question_index": 0,
                            "answer": (
                                "For empty dict inputs {}, normalize({}) returns {'empty': True}."
                            ),
                            "matched_slot_ids": ["S1"],
                        }
                    ]
                }
            )
        )

    outcome = run_swe_interact_e2e(
        task,
        agent_provider=MockLLMProvider(responder=agent_responder),
        oracle_provider=MockLLMProvider(responder=oracle_responder),
        mode="interactive",
        work_root=tmp_path / "run_ws",
        install_deps=False,
        cleanup=False,
    )
    assert outcome.passed is True
    assert outcome.base_passed is True
    assert outcome.new_passed is True
    assert outcome.test_patch_applied is True
    assert outcome.questions_asked == 1
    assert outcome.clarify_rounds == 1
    assert outcome.leakage_detected is False
    assert "{'empty': True}" in seen_worker_prompt[0]
    assert "normalize(data)" in outcome.model_patch
