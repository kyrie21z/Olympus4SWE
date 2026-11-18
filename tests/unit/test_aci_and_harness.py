"""Unit tests for Poka-Yoke ACI tools, git worktree sandbox, and AgentLoop."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from olympus.core.contracts import BudgetPolicy
from olympus.core.store import RunStore
from olympus.harness.loop import AgentLoop
from olympus.harness.worktree import capture_sandbox_diff, create_sandbox, remove_sandbox
from olympus.providers.base import (
    ChatMessage,
    LLMResponse,
    MockLLMProvider,
    ToolCall,
    ToolSpec,
    UsageStats,
)
from olympus.tools.aci import WorkspaceToolkit


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
