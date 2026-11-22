"""Unit tests for Poka-Yoke ACI tools, git worktree sandbox, and AgentLoop."""

from __future__ import annotations

import ast
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import olympus.tools.aci as aci_module
from olympus.core.contracts import BudgetPolicy
from olympus.core.store import RunStore
from olympus.harness.loop import AgentLoop
from olympus.harness.worktree import (
    absorb_verification_artifacts,
    capture_sandbox_diff,
    collect_sandbox_changes,
    create_sandbox,
    load_sandbox_baseline,
    remove_sandbox,
    save_sandbox_baseline,
    snapshot_sandbox_state,
    sync_upstream_files,
)
from olympus.providers.base import (
    ChatMessage,
    LLMResponse,
    MockLLMProvider,
    ToolCall,
    ToolSpec,
    UsageStats,
)
from olympus.tools.aci import (
    IGNORED_DIR_NAMES,
    WorkspacePolicy,
    WorkspaceToolkit,
    build_sandbox_env,
)


def test_aci_rejects_relative_and_escape_paths_and_enforces_write_scope(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    allowed_file = src_dir / "allowed.py"
    allowed_file.write_text("x = 1\n", encoding="utf-8")
    blocked_file = src_dir / "other.py"
    blocked_file.write_text("y = 2\n", encoding="utf-8")

    toolkit = WorkspaceToolkit(tmp_path, allowed_write_scope=["src/allowed.py"])

    # Relative path rejected with actionable hint containing workspace root
    rel_err = toolkit.execute("read_file", {"path": "src/allowed.py"})
    assert "Relative path" in rel_err
    assert str(tmp_path) in rel_err

    # Escape outside workspace rejected
    esc_err = toolkit.execute("read_file", {"path": "/etc/passwd"})
    assert "escapes workspace root" in esc_err

    # Out-of-scope write rejected
    scope_err = toolkit.execute(
        "write_file",
        {"path": str(blocked_file), "content": "y = 99\n"},
    )
    assert "outside allowed write_scope" in scope_err
    assert blocked_file.read_text(encoding="utf-8") == "y = 2\n"

    # In-scope edit succeeds
    ok = toolkit.execute(
        "edit_file",
        {"path": str(allowed_file), "target": "x = 1", "replacement": "x = 42"},
    )
    assert "1 replacement" in ok
    assert allowed_file.read_text(encoding="utf-8") == "x = 42\n"


def test_aci_read_only_hides_write_tools_and_truncates_with_hint(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"line-{i}" for i in range(500)), encoding="utf-8")

    ro = WorkspaceToolkit(tmp_path, read_only=True, max_output_chars=300)
    names = [t.name for t in ro.tool_specs()]
    assert names == ["glob_files", "grep_files", "read_file"]

    out = ro.execute("read_file", {"path": str(big), "start_line": 1, "end_line": 200})
    assert "[TRUNCATED:" in out
    assert "start_line=" in out


def test_git_worktree_sandbox_isolates_changes_and_captures_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    handle = create_sandbox(repo, label="task-1-a1")
    try:
        assert handle.is_git_worktree is True
        assert handle.base_commit is not None

        # Modify tracked + add untracked file inside sandbox only
        (handle.workspace_path / "app.py").write_text(
            "def f() -> int:\n    return 2\n", encoding="utf-8"
        )
        (handle.workspace_path / "new_mod.py").write_text("VAL = 9\n", encoding="utf-8")

        diff = capture_sandbox_diff(handle)
        assert "return 2" in diff
        assert "new_mod.py" in diff
        assert "VAL = 9" in diff
        # Main repo untouched
        assert "return 1" in (repo / "app.py").read_text(encoding="utf-8")
        assert not (repo / "new_mod.py").exists()
    finally:
        remove_sandbox(handle)
    assert not handle.workspace_path.exists()


def test_baseline_scoped_diff_only_emits_paths_mutated_since_baseline(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    handle = create_sandbox(repo, label="task-scope-a1")
    try:
        ws = handle.workspace_path
        # accepted_overlay-style files already exist before the pre-worker baseline
        (ws / "upstream.py").write_text("UPSTREAM = 1\n", encoding="utf-8")
        (ws / "stable.py").write_text("STABLE = True\n", encoding="utf-8")
        (ws / "old_name.py").write_text("RENAMED = True\n", encoding="utf-8")
        baseline = snapshot_sandbox_state(handle)

        (ws / "app.py").write_text("def f() -> int:\n    return 2\n", encoding="utf-8")
        (ws / "new_mod.py").write_text("VAL = 9\n", encoding="utf-8")
        (ws / "old_name.py").rename(ws / "renamed.py")
        (ws / "upstream.py").unlink()

        scoped = capture_sandbox_diff(handle, baseline)
        assert "return 2" in scoped
        assert "new_mod.py" in scoped and "VAL = 9" in scoped
        assert "rename from old_name.py" in scoped
        assert "rename to renamed.py" in scoped
        assert "deleted file mode" in scoped
        assert "upstream.py" in scoped
        assert "stable.py" not in scoped
        assert "STABLE = True" not in scoped

        unscoped = capture_sandbox_diff(handle)
        assert "stable.py" in unscoped and "STABLE = True" in unscoped
    finally:
        remove_sandbox(handle)


def _init_git_repo(root: Path) -> Path:
    repo = (root / "repo").resolve()
    repo.mkdir()
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def test_build_sandbox_env_isolates_temp_dirs_outside_the_git_worktree(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path)
    handle = create_sandbox(repo, label="task-tmp")
    expected = repo / ".olympus" / "sandbox_tmp" / "task-tmp"
    try:
        envs = (
            build_sandbox_env(handle.workspace_path, host_root=repo),
            build_sandbox_env(handle.workspace_path),
        )
        for env in envs:
            for name in ("TMPDIR", "TEMP", "TMP", "SQLITE_TMPDIR"):
                assert env[name] == str(expected)
            assert expected.is_dir()
            assert not expected.is_relative_to(handle.workspace_path)
    finally:
        remove_sandbox(handle)
    assert not expected.exists()
    assert not handle.workspace_path.exists()


def test_retained_sandbox_reuses_work_and_original_baseline(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    first = create_sandbox(repo, label="task-retain")
    assert first.reused is False
    baseline = snapshot_sandbox_state(first)
    save_sandbox_baseline(first, baseline)
    (first.workspace_path / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
    (first.workspace_path / "scratch.db").write_text("stray\n", encoding="utf-8")
    try:
        reused = create_sandbox(repo, label="task-retain", reuse=True)
        assert reused.reused is True
        assert reused.workspace_path == first.workspace_path
        assert (reused.workspace_path / "feature.py").read_text(encoding="utf-8") == "FLAG = True\n"
        assert load_sandbox_baseline(reused) == baseline
        assert collect_sandbox_changes(reused, baseline) == ["feature.py", "scratch.db"]
        remove_sandbox(reused)
        assert not first.workspace_path.exists()
        assert load_sandbox_baseline(first) is None

        fresh = create_sandbox(repo, label="task-retain")
        try:
            assert fresh.reused is False
            assert not (fresh.workspace_path / "feature.py").exists()
        finally:
            remove_sandbox(fresh)
    finally:
        if first.workspace_path.exists():
            remove_sandbox(first)


def test_sync_upstream_files_updates_only_synced_baseline_entries(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    handle = create_sandbox(repo, label="task-sync")
    try:
        ws = handle.workspace_path
        (ws / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
        baseline = snapshot_sandbox_state(handle)
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        overlay = tmp_path / "overlay"
        overlay.mkdir()
        (overlay / "shared.py").write_text("VALUE = 2\n", encoding="utf-8")
        (overlay / "feature.py").write_text("UPSTREAM = True\n", encoding="utf-8")
        (overlay / "new_upstream.py").write_text("NEW = True\n", encoding="utf-8")

        synced = sync_upstream_files(handle, overlay, baseline, write_scope=["feature.py"])

        assert (ws / "shared.py").read_text(encoding="utf-8") == "VALUE = 2\n"
        assert (ws / "feature.py").read_text(encoding="utf-8") == "FLAG = True\n"
        assert (ws / "new_upstream.py").read_text(encoding="utf-8") == "NEW = True\n"
        assert collect_sandbox_changes(handle, synced) == ["feature.py"]
        assert collect_sandbox_changes(handle, baseline) == [
            "feature.py",
            "new_upstream.py",
            "shared.py",
        ]
    finally:
        remove_sandbox(handle)


def test_absorb_verification_artifacts_hides_only_out_of_scope_side_effects(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "contract.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "contract.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add contract"], cwd=repo, check=True)
    handle = create_sandbox(repo, label="task-checks")
    try:
        ws = handle.workspace_path
        baseline = snapshot_sandbox_state(handle)
        (ws / "feature.py").write_text("FLAG = True\n", encoding="utf-8")
        (ws / "check-ran.txt").write_text("ran\n", encoding="utf-8")
        (ws / "contract.txt").write_text("", encoding="utf-8")

        absorbed = absorb_verification_artifacts(handle, baseline, write_scope=["feature.py"])

        assert collect_sandbox_changes(handle, absorbed) == ["contract.txt", "feature.py"]
    finally:
        remove_sandbox(handle)


def test_sync_upstream_files_never_overwrites_worker_deletion_of_tracked_file(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "upstream.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "upstream.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add upstream"], cwd=repo, check=True)
    handle = create_sandbox(repo, label="task-del")
    try:
        ws = handle.workspace_path
        baseline = snapshot_sandbox_state(handle)
        save_sandbox_baseline(handle, baseline)
        (ws / "upstream.txt").unlink()
        overlay = tmp_path / "overlay"
        overlay.mkdir()
        (overlay / "upstream.txt").write_text("accepted later\n", encoding="utf-8")

        reopened = create_sandbox(repo, label="task-del", reuse=True)
        loaded = load_sandbox_baseline(reopened)
        assert loaded is not None
        synced = sync_upstream_files(reopened, overlay, loaded, write_scope=["feature.py"])

        assert not (ws / "upstream.txt").exists()
        assert collect_sandbox_changes(reopened, synced) == ["upstream.txt"]
    finally:
        remove_sandbox(handle)


def test_capture_sandbox_diff_handles_nongit_binary_and_reverted_overlay(
    tmp_path: Path,
) -> None:
    nongit_root = tmp_path / "plain"
    nongit_root.mkdir()
    nongit_handle = create_sandbox(nongit_root, label="plain-diff")
    try:
        nongit_baseline = snapshot_sandbox_state(nongit_handle)
        (nongit_handle.workspace_path / "feature.txt").write_text("hello\n", encoding="utf-8")
        nongit_diff = capture_sandbox_diff(nongit_handle, nongit_baseline)
        assert "feature.txt" in nongit_diff
        assert "+hello" in nongit_diff
    finally:
        remove_sandbox(nongit_handle)

    repo = _init_git_repo(tmp_path)
    (repo / "tracked.txt").write_text("HEAD\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add tracked"], cwd=repo, check=True)
    handle = create_sandbox(repo, label="git-diff-edge")
    try:
        ws = handle.workspace_path
        (ws / "tracked.txt").write_text("accepted overlay\n", encoding="utf-8")
        baseline = snapshot_sandbox_state(handle)
        (ws / "tracked.txt").write_text("HEAD\n", encoding="utf-8")
        (ws / "binary.dat").write_bytes(b"\x00\xff\x01")
        (ws / "added.txt").write_text("hello\n", encoding="utf-8")

        diff = capture_sandbox_diff(handle, baseline)
        assert "tracked.txt" in diff
        assert "GIT binary patch" in diff
        assert "@@ -0,0 +1 @@" in diff
    finally:
        remove_sandbox(handle)


def test_agent_loop_preserves_prefix_and_errors_and_respects_budget(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "hello.txt"
    target.write_text("hello world\n", encoding="utf-8")

    toolkit = WorkspaceToolkit(ws, allowed_write_scope=["hello.txt"])
    store = RunStore(tmp_path / "state.db")
    run_id = store.create_run(title="Loop test", budget_policy=BudgetPolicy(max_llm_calls=5))
    trace_path = tmp_path / "trace.jsonl"

    # Step 1: intentional bad relative path -> error stays in context
    # Step 2: corrected absolute path -> succeeds
    # Step 3: final answer
    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Try reading file",
                tool_calls=(
                    ToolCall(id="tc-1", name="read_file", arguments={"path": "hello.txt"}),
                ),
                usage=UsageStats(input_tokens=100, output_tokens=20),
            ),
            LLMResponse(
                content="Use absolute path instead",
                tool_calls=(
                    ToolCall(id="tc-2", name="read_file", arguments={"path": str(target)}),
                ),
                usage=UsageStats(input_tokens=140, output_tokens=25),
            ),
            LLMResponse(
                content="Done reading hello world.",
                tool_calls=(),
                usage=UsageStats(input_tokens=180, output_tokens=15),
            ),
        ]
    )

    loop = AgentLoop(
        provider=provider,
        toolkit=toolkit,
        store=store,
        run_id=run_id,
        actor="hephaestus",
        max_steps=5,
        trace_path=trace_path,
    )
    res = loop.run(system_prompt="SYS", user_prompt="Read hello.txt")

    assert res.completed is True
    assert res.steps_taken == 3
    assert res.final_content == "Done reading hello world."
    assert res.total_input_tokens == 420
    assert res.total_output_tokens == 60

    # Verify exact prefix invariance across every LLM call (KV-cache property)
    req0_msgs, req0_tools = provider.recorded_requests[0]
    req1_msgs, req1_tools = provider.recorded_requests[1]
    req2_msgs, req2_tools = provider.recorded_requests[2]
    assert req0_tools == req1_tools == req2_tools
    assert req1_msgs[: len(req0_msgs)] == req0_msgs
    assert req2_msgs[: len(req1_msgs)] == req1_msgs
    # Failed tool output preserved in messages
    assert "Relative path" in req1_msgs[-1].content

    # Verify trace.jsonl and monotonic budget ledger
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 3
    assert len({t["prefix_hash"] for t in traces}) == 1
    snap = store.get_run(run_id)
    assert snap["budget_ledger"].llm_calls_used == 3


def test_agent_loop_stops_on_budget_exhaustion_and_max_steps(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state.db")
    run_id = store.create_run(title="Budget stop", budget_policy=BudgetPolicy(max_llm_calls=1))

    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Call tool",
                tool_calls=(ToolCall(id="tc-1", name="read_file", arguments={"path": "/missing"}),),
            ),
            LLMResponse(content="Should not reach"),
        ]
    )
    loop = AgentLoop(provider=provider, store=store, run_id=run_id, max_steps=5)
    res = loop.run(system_prompt="SYS", user_prompt="Go")
    assert res.completed is False
    assert res.stop_reason.startswith("budget_exhausted")
    assert res.steps_taken == 1

    # Also check max_steps stopping condition independent of budget
    loop2 = AgentLoop(
        provider=MockLLMProvider(
            responses=[
                LLMResponse(
                    content="step1",
                    tool_calls=(ToolCall(id="tc-1", name="read_file", arguments={"path": "/a"}),),
                ),
                LLMResponse(
                    content="step2",
                    tool_calls=(ToolCall(id="tc-2", name="read_file", arguments={"path": "/b"}),),
                ),
            ]
        ),
        max_steps=2,
    )
    res2 = loop2.run(system_prompt="SYS", user_prompt="Go")
    assert res2.completed is False
    assert res2.stop_reason == "max_steps_exceeded"
    assert res2.steps_taken == 2


def test_agent_loop_synthesizes_final_answer_when_enabled(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    seen_tool_counts: list[int] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages
        seen_tool_counts.append(len(tools))
        if len(seen_tool_counts) == 1:
            return LLMResponse(
                content="reading",
                tool_calls=(
                    ToolCall(
                        id="tc-1",
                        name="read_file",
                        arguments={"path": str(tmp_path / "mod.py")},
                    ),
                ),
            )
        return LLMResponse(content="synthesized answer")

    loop = AgentLoop(
        provider=MockLLMProvider(responder=responder),
        toolkit=WorkspaceToolkit(tmp_path, read_only=True),
        max_steps=2,
        synthesize_on_max_steps=True,
    )
    res = loop.run(system_prompt="SYS", user_prompt="Go")
    assert res.completed is True
    assert res.final_content == "synthesized answer"
    assert seen_tool_counts[0] > 0
    assert seen_tool_counts[1] == 0


def test_run_structured_loop_repairs_invalid_json_without_reexploring(tmp_path: Path) -> None:
    from pydantic import Field

    from olympus.core.contracts import StrictModel
    from olympus.stages.common import run_structured_loop

    class Out(StrictModel):
        value: int = Field(ge=1)

    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    tool_counts: list[int] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        tool_counts.append(len(tools))
        if len(tool_counts) == 1:
            return LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="tc-1",
                        name="read_file",
                        arguments={"path": str(tmp_path / "mod.py")},
                    ),
                ),
            )
        if len(tool_counts) == 2:
            return LLMResponse(content=json.dumps({"value": 0}))
        assert len(messages) >= 5
        assert "mod.py" in messages[2].content or "x = 1" in messages[3].content
        return LLMResponse(content=json.dumps({"value": 7}))

    loop = AgentLoop(
        provider=MockLLMProvider(responder=responder),
        toolkit=WorkspaceToolkit(tmp_path, read_only=True),
        max_steps=4,
    )
    out = run_structured_loop(
        loop=loop,
        system_prompt="SYS",
        user_prompt="Go",
        model_cls=Out,
    )
    assert out.value == 7
    assert tool_counts[0] > 0 and tool_counts[1] > 0
    assert tool_counts[2] == 0


def test_agent_loop_retries_once_on_transient_timeout() -> None:
    calls = 0

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal calls
        del messages, tools
        calls += 1
        if calls == 1:
            raise TimeoutError("The read operation timed out")
        return LLMResponse(content="recovered")

    loop = AgentLoop(provider=MockLLMProvider(responder=responder), max_steps=2)
    res = loop.run(system_prompt="SYS", user_prompt="Go")
    assert res.completed is True
    assert res.final_content == "recovered"
    assert calls == 2


def test_run_structured_loop_salvages_submit_tool_and_recovers_max_steps(tmp_path: Path) -> None:
    from pydantic import Field

    from olympus.core.contracts import StrictModel
    from olympus.stages.common import run_structured_loop

    class Out(StrictModel):
        value: int = Field(ge=1)

    # 1. Hallucinated submit_plan tool call on max_steps is salvaged without extra LLM calls.
    def submit_responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages, tools
        return LLMResponse(
            content="Here is the plan:",
            tool_calls=(
                ToolCall(
                    id="sp-1",
                    name="submit_plan",
                    arguments={"plan": json.dumps({"value": 9})},
                ),
            ),
        )

    loop1 = AgentLoop(
        provider=MockLLMProvider(responder=submit_responder),
        toolkit=WorkspaceToolkit(tmp_path, read_only=True),
        max_steps=1,
    )
    out1 = run_structured_loop(loop=loop1, system_prompt="S", user_prompt="U", model_cls=Out)
    assert out1.value == 9

    # 2. Empty first repair after max_steps retries cleanly with max_parse_retries=2 default.
    turns = 0

    def repair_responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal turns
        del messages, tools
        turns += 1
        if turns == 1:
            return LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="g-1",
                        name="glob_files",
                        arguments={"path": str(tmp_path), "pattern": "*"},
                    ),
                ),
            )
        if turns == 2:
            return LLMResponse(content="")
        return LLMResponse(content=json.dumps({"value": 11}))

    loop2 = AgentLoop(
        provider=MockLLMProvider(responder=repair_responder),
        toolkit=WorkspaceToolkit(tmp_path, read_only=True),
        max_steps=1,
    )
    out2 = run_structured_loop(loop=loop2, system_prompt="S", user_prompt="U", model_cls=Out)
    assert out2.value == 11
    assert turns == 3


def test_run_structured_forces_final_step_synthesis_without_mutating_flag(
    tmp_path: Path,
) -> None:
    from pydantic import Field

    from olympus.core.contracts import StrictModel

    class Out(StrictModel):
        value: int = Field(ge=1)

    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    seen_tools: list[int] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages
        seen_tools.append(len(tools))
        if len(seen_tools) == 1:
            return LLMResponse(
                content="reading",
                tool_calls=(
                    ToolCall(
                        id="tc-1",
                        name="read_file",
                        arguments={"path": str(tmp_path / "mod.py")},
                    ),
                ),
            )
        return LLMResponse(content=json.dumps({"value": 3}))

    loop = AgentLoop(
        provider=MockLLMProvider(responder=responder),
        toolkit=WorkspaceToolkit(tmp_path, read_only=True),
        max_steps=2,
    )
    assert loop.synthesize_on_max_steps is False
    out = loop.run_structured(system_prompt="SYS", user_prompt="GO", model_cls=Out)

    assert out.value == 3
    # Synthesis is requested per call; the loop-level flag must stay untouched.
    assert loop.synthesize_on_max_steps is False
    assert seen_tools == [3, 0]


def test_run_structured_retries_domain_validation_with_feedback() -> None:
    from pydantic import Field

    from olympus.core.contracts import StrictModel

    class Out(StrictModel):
        value: int = Field(ge=1)

    prompts: list[str] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        prompts.append(messages[-1].content)
        return LLMResponse(content=json.dumps({"value": 1 if len(prompts) == 1 else 5}))

    def validator(candidate: Out) -> Sequence[str]:
        return [] if candidate.value >= 5 else [f"value {candidate.value} is below 5"]

    loop = AgentLoop(provider=MockLLMProvider(responder=responder), max_steps=1)
    out = loop.run_structured(
        system_prompt="SYS",
        user_prompt="BASE PROMPT",
        model_cls=Out,
        max_validation_retries=2,
        validation_header="[DOMAIN FAILED]",
        validation_instruction="Return at least 5.",
        validator=validator,
    )

    assert out.value == 5
    assert len(prompts) == 2
    assert prompts[0] == "BASE PROMPT"
    assert "[DOMAIN FAILED]" in prompts[1]
    assert "value 1 is below 5" in prompts[1]
    assert "Return at least 5." in prompts[1]


def test_run_structured_applies_transform_before_validation() -> None:
    from pydantic import Field

    from olympus.core.contracts import StrictModel

    class Bound(StrictModel):
        value: int = Field(ge=1)
        spec_id: str | None = None

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages, tools
        return LLMResponse(content=json.dumps({"value": 2}))

    seen_spec_ids: list[str | None] = []

    def validator(candidate: Bound) -> Sequence[str]:
        seen_spec_ids.append(candidate.spec_id)
        return [] if candidate.spec_id == "spec-1" else ["missing spec binding"]

    loop = AgentLoop(provider=MockLLMProvider(responder=responder), max_steps=1)
    out = loop.run_structured(
        system_prompt="SYS",
        user_prompt="GO",
        model_cls=Bound,
        transform=lambda candidate: candidate.model_copy(update={"spec_id": "spec-1"}),
        validator=validator,
    )

    assert seen_spec_ids == ["spec-1"]
    assert out.spec_id == "spec-1"


def test_run_structured_invokes_on_validation_exhausted() -> None:
    from pydantic import Field

    from olympus.core.contracts import StrictModel

    class Out(StrictModel):
        value: int = Field(ge=1)

    calls = 0

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal calls
        del messages, tools
        calls += 1
        return LLMResponse(content=json.dumps({"value": 42}))

    exhausted: list[tuple[int, list[str]]] = []

    def on_exhausted(candidate: Out, violations: list[str]) -> Out:
        exhausted.append((candidate.value, violations))
        return candidate.model_copy(update={"value": 7})

    loop = AgentLoop(provider=MockLLMProvider(responder=responder), max_steps=1)
    out = loop.run_structured(
        system_prompt="SYS",
        user_prompt="GO",
        model_cls=Out,
        max_validation_retries=1,
        validator=lambda candidate: ["value is forbidden"],
        on_validation_exhausted=on_exhausted,
    )

    assert calls == 2
    assert exhausted == [(42, ["value is forbidden"])]
    assert out.value == 7


def test_workspace_policy_authorizes_posix_scopes_and_rejects_escapes(tmp_path: Path) -> None:
    workspace = (tmp_path / "ws").resolve()
    workspace.mkdir()

    policy = WorkspacePolicy(
        workspace_root=workspace,
        allowed_write_scope=["src/allowed.py", "docs/", "pkg/*"],
    )

    # Absolute paths inside the workspace root normalize to repo-relative form.
    assert policy.authorize_tool("write_file", {"path": str(workspace / "src/allowed.py")}) == (
        True,
        "",
    )
    # Relative and "./"-prefixed paths cover directory prefixes and globs.
    assert policy.authorize_tool("write_file", {"path": "./docs/nested/x.md"}) == (True, "")
    assert policy.authorize_tool("write_file", {"path": "pkg/sub/mod.py"}) == (True, "")
    assert policy.authorize_tool("read_file", {"path": "anything/outside.txt"}) == (True, "")

    allowed, reason = policy.authorize_tool("write_file", {"path": "src/other.py"})
    assert allowed is False
    assert "outside authorized write_scope" in reason

    allowed, reason = policy.authorize_tool("write_file", {"path": "/etc/passwd"})
    assert allowed is False
    assert "escapes workspace root" in reason

    allowed, reason = policy.authorize_tool("write_file", {"path": "../escape.py"})
    assert allowed is False
    assert ".." in reason

    # Symlink escapes are rejected for both absolute and relative spellings.
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    for raw in (str(workspace / "link" / "escape.txt"), "link/escape.txt"):
        allowed, reason = policy.authorize_tool("write_file", {"path": raw})
        assert allowed is False
        assert "escapes workspace root" in reason


def test_workspace_policy_forbidden_scope_blocks_shared_contracts(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    policy = WorkspacePolicy(
        workspace_root=workspace,
        allowed_write_scope=["src/"],
        forbidden_write_scope=["src/contracts/", "src/generated/*.json"],
    )
    assert policy.authorize_tool("write_file", {"path": "src/api/app.py"}) == (True, "")
    for target in ("src/contracts/dto.py", str(workspace / "src/contracts/dto.py")):
        allowed, reason = policy.authorize_tool("write_file", {"path": target})
        assert allowed is False
        assert "read-only shared contract path" in reason
    allowed, reason = policy.authorize_tool("write_file", {"path": "src/generated/schema.json"})
    assert allowed is False
    assert "read-only shared contract path" in reason


def test_workspace_policy_gates_run_bash_on_writable_scope(tmp_path: Path) -> None:
    assert frozenset({"read_file", "glob_files", "grep_files"}) == WorkspacePolicy.READ_ONLY_TOOLS
    assert "run_bash" not in WorkspacePolicy.READ_ONLY_TOOLS
    assert frozenset({"write_file", "edit_file"}) == WorkspacePolicy.WRITE_TOOLS

    denied = WorkspacePolicy(workspace_root=tmp_path, allowed_write_scope=[])
    assert denied.authorize_tool("run_bash", {"command": "ls"}) == (
        False,
        "No write_scope authorized for this worker.",
    )
    assert denied.authorize_tool("write_file", {"path": "x.py"}) == (
        False,
        "No write_scope authorized for this worker.",
    )
    assert denied.authorize_tool("grep_files", {"pattern": "x"}) == (True, "")

    read_only = WorkspacePolicy(
        workspace_root=tmp_path, read_only=True, allowed_write_scope=["src/"]
    )
    allowed, reason = read_only.authorize_tool("run_bash", {"command": "ls"})
    assert allowed is False
    assert "read-only" in reason
    assert read_only.authorize_tool("read_file", {"path": "src/x.py"}) == (True, "")

    assert WorkspacePolicy(
        workspace_root=tmp_path, allowed_write_scope=["src/"]
    ).authorize_tool("run_bash", {"command": "ls"}) == (True, "")
    assert WorkspacePolicy(workspace_root=tmp_path).authorize_tool(
        "run_bash", {"command": "ls"}
    ) == (True, "")

    unknown = WorkspacePolicy(workspace_root=tmp_path, allowed_write_scope=["src/"])
    assert unknown.authorize_tool("submit_plan", {}) == (
        False,
        "Tool 'submit_plan' is not authorized for this worker.",
    )


def test_workspace_toolkit_shares_one_policy_with_its_hook(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    toolkit = WorkspaceToolkit(
        workspace,
        allowed_write_scope=["docs/"],
        forbidden_write_scope=["docs/locked/"],
    )
    assert toolkit.policy.workspace_root == workspace
    assert toolkit.policy.allowed_write_scope == ("docs/",)
    assert toolkit.policy.forbidden_write_scope == ("docs/locked/",)

    hook = toolkit.as_pre_tool_hook()
    assert hook("write_file", {"path": "docs/guide.md"}) == (True, "")
    allowed, reason = hook("write_file", {"path": "docs/locked/secret.md"})
    assert allowed is False
    assert "read-only shared contract path" in reason

    # The handler enforces the exact same policy, not only the loop hook.
    err = toolkit.execute(
        "write_file", {"path": str(workspace / "docs/locked/secret.md"), "content": "x"}
    )
    assert "read-only shared contract path" in err
    assert not (workspace / "docs/locked/secret.md").exists()

    ok = toolkit.execute(
        "write_file", {"path": str(workspace / "docs/guide.md"), "content": "hello"}
    )
    assert "Wrote 5 chars" in ok
    assert (workspace / "docs/guide.md").read_text(encoding="utf-8") == "hello"


def test_workspace_toolkit_uses_injected_env_builder(tmp_path: Path) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake_env(workspace: Path, env_root: Path) -> dict[str, str]:
        calls.append((workspace, env_root))
        return dict(os.environ)

    workspace = tmp_path.resolve()
    toolkit = WorkspaceToolkit(workspace, env_builder=fake_env)
    out = toolkit.execute("run_bash", {"command": "printf ok"})
    assert "ok" in out
    assert calls == [(workspace, workspace)]


def test_tools_aci_has_no_harness_imports_and_worktree_reexports_env_builder() -> None:
    aci_path = Path(aci_module.__file__).resolve()
    tree = ast.parse(aci_path.read_text(encoding="utf-8"))
    harness_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("olympus.harness"):
            harness_imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            harness_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("olympus.harness")
            )
    assert harness_imports == []

    from olympus.harness.worktree import build_sandbox_env as worktree_env

    assert worktree_env is build_sandbox_env
    assert ".olympus" in IGNORED_DIR_NAMES
