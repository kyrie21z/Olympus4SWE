"""Unit tests for Waku runtime primitives: trimming, hooks, observers, and receipts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from olympus.core.contracts import (
    AcceptanceCheck,
    BudgetPolicy,
    CheckKind,
    ReviewVerdict,
    TaskNode,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.harness.hooks import default_write_scope_hook
from olympus.harness.loop import AgentLoop
from olympus.harness.observer import compose_observers
from olympus.harness.receipt import RECEIPT_KEYS, _format_k, build_turn_receipt
from olympus.harness.trim import KEEP_CHARS, LONG_CHARS, TRIM_NOTE, chain, shrink_seen
from olympus.providers.base import (
    ChatMessage,
    LLMResponse,
    MockLLMProvider,
    ToolCall,
    ToolSpec,
    UsageStats,
)
from olympus.stages.orchestrate import _write_manifest_md
from olympus.tools.aci import WorkspaceToolkit


def _assistant(content: str = "thinking") -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


def test_shrink_seen_trims_only_superseded_long_tool_outputs() -> None:
    stale = "A" * 5000
    newest = "B" * 5000
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="go"),
        _assistant(),
        ChatMessage(role="tool", content=stale, tool_call_id="t1", name="read_file"),
        _assistant(),
        ChatMessage(role="tool", content=newest, tool_call_id="t2", name="read_file"),
    ]

    shrunk = shrink_seen(messages)

    trimmed = shrunk[3]
    assert trimmed.content.startswith("A" * KEEP_CHARS)
    assert TRIM_NOTE.format(total=5000) in trimmed.content
    assert len(trimmed.content) == KEEP_CHARS + len(TRIM_NOTE.format(total=5000))
    # Newest tool output (after the last assistant turn) stays byte-for-byte intact
    assert shrunk[5].content == newest
    # Short and non-tool messages are never touched
    assert shrunk[0] is messages[0]
    assert shrunk[2] == messages[2]
    # Idempotent: an already-trimmed output is left alone
    assert shrink_seen(shrunk)[3] == trimmed


def test_shrink_seen_without_assistant_returns_copy() -> None:
    messages = [
        ChatMessage(role="user", content="go"),
        ChatMessage(role="tool", content="Z" * 5000, tool_call_id="t1", name="read_file"),
    ]
    shrunk = shrink_seen(messages)
    assert shrunk == messages
    assert shrunk is not messages


def test_chain_composes_transforms_left_to_right() -> None:
    def append_note(items: list[ChatMessage]) -> list[ChatMessage]:
        return [*items, ChatMessage(role="user", content="note")]

    def uppercase_last(items: list[ChatMessage]) -> list[ChatMessage]:
        return [*items[:-1], ChatMessage(role="user", content=items[-1].content.upper())]

    pipeline = chain(append_note, uppercase_last)
    out = pipeline([ChatMessage(role="user", content="start")])
    assert [m.content for m in out] == ["start", "NOTE"]


def test_agent_loop_trims_seen_tool_output_before_third_step(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    big = ws / "big.txt"
    big.write_text("\n".join("x" * 60 for _ in range(150)), encoding="utf-8")
    small = ws / "small.txt"
    small.write_text("tiny\n", encoding="utf-8")
    seen: list[list[ChatMessage]] = []

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        seen.append(list(messages))
        if len(seen) == 1:
            return LLMResponse(
                content="read big",
                tool_calls=(
                    ToolCall(id="t1", name="read_file", arguments={"path": str(big)}),
                ),
            )
        if len(seen) == 2:
            return LLMResponse(
                content="read small",
                tool_calls=(
                    ToolCall(id="t2", name="read_file", arguments={"path": str(small)}),
                ),
            )
        return LLMResponse(content="done")

    loop = AgentLoop(
        provider=MockLLMProvider(responder=responder),
        toolkit=WorkspaceToolkit(ws, read_only=True),
        max_steps=5,
    )
    res = loop.run(system_prompt="SYS", user_prompt="GO")

    assert res.completed is True
    assert res.steps_taken == 3

    raw_first = next(m.content for m in seen[1] if m.role == "tool")
    assert len(raw_first) > LONG_CHARS
    assert "[Cut from " not in raw_first

    tool_messages = [m for m in seen[2] if m.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0].content == raw_first[:KEEP_CHARS] + TRIM_NOTE.format(
        total=len(raw_first)
    )
    assert len(tool_messages[0].content) < len(raw_first)
    # The freshest evidence is untouched
    assert "tiny" in tool_messages[1].content
    assert "[Cut from " not in tool_messages[1].content
    assert res.receipt["turn_id"].startswith("turn-")


def test_default_write_scope_hook_matches_globs_and_denies_empty_scope(tmp_path: Path) -> None:
    hook = default_write_scope_hook(["src/waitlist/*", "docs/"], workspace_root=tmp_path)

    assert hook("read_file", {"path": str(tmp_path / "anything.txt")}) == (True, "")
    assert hook("glob_files", {"pattern": "**/*.py"}) == (True, "")
    assert hook("grep_files", {"pattern": "class"}) == (True, "")
    assert hook("run_bash", {"command": "ls"}) == (True, "")
    assert hook("write_file", {"path": str(tmp_path / "src/waitlist/a.py")}) == (True, "")
    # Relative paths are normalized by stripping the leading ./
    assert hook("edit_file", {"path": "./src/waitlist/a.py"}) == (True, "")
    # "docs/" is normalized to "docs/*"
    assert hook("write_file", {"path": str(tmp_path / "docs/readme.md")}) == (True, "")

    allowed, reason = hook("write_file", {"path": str(tmp_path / "src/other.py")})
    assert allowed is False
    assert "outside authorized write_scope" in reason

    # Absolute paths escaping the workspace stay rejected
    assert hook("write_file", {"path": "/etc/passwd"})[0] is False

    exact = default_write_scope_hook(["src/waitlist/waitlist.py"], workspace_root=tmp_path)
    assert exact("write_file", {"path": "src/waitlist/waitlist.py"}) == (True, "")
    assert exact("write_file", {"path": "src/waitlist/other.py"})[0] is False

    empty = default_write_scope_hook([], workspace_root=tmp_path)
    assert empty("read_file", {"path": str(tmp_path)}) == (True, "")
    assert empty("write_file", {"path": str(tmp_path / "x.py")}) == (
        False,
        "No write_scope authorized for this worker.",
    )


def test_agent_loop_blocks_unauthorized_write_before_execution(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    allowed_file = ws / "allowed.py"
    allowed_file.write_text("x = 1\n", encoding="utf-8")
    blocked_file = ws / "blocked.py"
    events: list[tuple[str, dict[str, Any]]] = []

    def observer(kind: str, payload: dict[str, Any]) -> None:
        events.append((kind, payload))

    blocked_provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="write outside scope",
                tool_calls=(
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": str(blocked_file), "content": "y = 2\n"},
                    ),
                ),
                usage=UsageStats(input_tokens=100, output_tokens=20),
            ),
            LLMResponse(
                content="acknowledged",
                usage=UsageStats(input_tokens=120, output_tokens=5),
            ),
        ]
    )
    blocked_loop = AgentLoop(
        provider=blocked_provider,
        toolkit=WorkspaceToolkit(ws),
        observer=observer,
        pre_tool_hook=default_write_scope_hook(["allowed.py"], workspace_root=ws),
        task_id="T1",
        max_steps=3,
    )
    blocked_res = blocked_loop.run(system_prompt="SYS", user_prompt="GO")

    assert blocked_res.completed is True
    assert blocked_file.exists() is False
    denied = [payload for kind, payload in events if kind == "pre_tool_blocked"]
    assert len(denied) == 1
    assert denied[0]["tool"] == "write_file"
    assert denied[0]["task_id"] == "T1"
    assert "outside authorized write_scope" in denied[0]["reason"]
    denied_outputs = [msg.content for msg in blocked_res.messages if msg.role == "tool"]
    assert any("[PRE_TOOL_HOOK DENIED]" in content for content in denied_outputs)
    assert [entry["status"] for entry in blocked_res.receipt["tools"]] == ["denied"]

    # An in-scope write still executes normally through the same hook
    allowed_provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="write inside scope",
                tool_calls=(
                    ToolCall(
                        id="w2",
                        name="write_file",
                        arguments={"path": str(allowed_file), "content": "x = 2\n"},
                    ),
                ),
            ),
            LLMResponse(content="done"),
        ]
    )
    allowed_loop = AgentLoop(
        provider=allowed_provider,
        toolkit=WorkspaceToolkit(ws),
        pre_tool_hook=default_write_scope_hook(["allowed.py"], workspace_root=ws),
        max_steps=3,
    )
    allowed_res = allowed_loop.run(system_prompt="SYS", user_prompt="GO")
    assert allowed_res.completed is True
    assert allowed_file.read_text(encoding="utf-8") == "x = 2\n"
    assert [entry["status"] for entry in allowed_res.receipt["tools"]] == ["ok"]


def test_compose_observers_swallows_external_ui_failures() -> None:
    received: list[str] = []

    def broken(kind: str, payload: dict[str, Any]) -> None:
        del kind, payload
        raise RuntimeError("ui observer exploded")

    def recorder(kind: str, payload: dict[str, Any]) -> None:
        del payload
        received.append(kind)

    observer = compose_observers(broken, None, recorder)
    observer("step_start", {"step": 1})
    observer("receipt", {"one_line": "x"})
    assert received == ["step_start", "receipt"]


def test_build_turn_receipt_aggregates_sanitizes_and_collapses() -> None:
    events: list[dict[str, Any]] = [
        {"kind": "llm", "input_tokens": 5700, "output_tokens": 120},
        {"kind": "step_start", "step": 1},
        {"kind": "tool_end", "tool": "read_file", "is_error": False, "duration_ms": 3},
        {"kind": "tool_end", "tool": "read_file", "is_error": True, "duration_ms": 4},
        {"kind": "pre_tool_blocked", "tool": "bad tool name!!", "reason": "nope"},
        {"kind": "tool_end", "tool": "run_bash", "duration_ms": 9},
        {"kind": "tool_end", "tool": "run_bash", "duration_ms": 10},
        {"kind": "tool_end", "tool": "read_file", "output": "SECRET_TOOL_OUTPUT"},
        {"kind": "llm", "calls": 2, "input_tokens": 300, "output_tokens": 30},
    ]

    receipt = build_turn_receipt(
        turn_id="turn-abc",
        actor="hephaestus:T1",
        task_id="T1",
        events=events,
        diff_lines=(12, 3),
        checks_passed=2,
        checks_total=3,
    )

    assert tuple(receipt) == RECEIPT_KEYS
    assert receipt["llm"] == {"calls": 3, "input_tokens": 6000, "output_tokens": 150}
    assert receipt["tools"][2] == {"tool": "unknown", "status": "denied", "duration_ms": 0}
    assert receipt["artifacts"] == {
        "diff_added": 12,
        "diff_removed": 3,
        "checks_passed": 2,
        "checks_total": 3,
    }
    assert receipt["one_line"] == (
        "hephaestus:T1 · 3 calls · 6.0k in / 150 out"
        " | read_file ×2 · unknown · run_bash ×2 · read_file | diff +12/-3"
    )
    assert "SECRET_TOOL_OUTPUT" not in json.dumps(receipt)


def test_format_k_thresholds() -> None:
    assert _format_k(0) == "0"
    assert _format_k(999) == "999"
    assert _format_k(2000) == "2.0k"
    assert _format_k(5700) == "5.7k"


def test_loop_emits_events_and_persists_sanitized_receipt(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    store = RunStore(tmp_path / "state.db")
    run_id = store.create_run(title="Receipt", budget_policy=BudgetPolicy(max_llm_calls=10))
    kinds: list[str] = []
    captured: list[dict[str, Any]] = []

    def observer(kind: str, payload: dict[str, Any]) -> None:
        kinds.append(kind)
        captured.append(payload)

    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="SECRET_ASSISTANT_TEXT",
                tool_calls=(
                    ToolCall(id="t1", name="read_file", arguments={"path": str(target)}),
                ),
                usage=UsageStats(input_tokens=3100, output_tokens=210),
            ),
            LLMResponse(
                content="SECRET_FINAL_ANSWER",
                usage=UsageStats(input_tokens=1200, output_tokens=90),
            ),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        toolkit=WorkspaceToolkit(ws, read_only=True),
        store=store,
        run_id=run_id,
        actor="hephaestus:T1",
        observer=observer,
        task_id="T1",
        max_steps=4,
    )
    res = loop.run(system_prompt="SYSTEM_SECRET_TOKEN", user_prompt="USER_SECRET_TOKEN")

    assert res.completed is True
    assert kinds == [
        "step_start",
        "llm",
        "tool_start",
        "tool_end",
        "step_start",
        "llm",
        "receipt",
    ]
    receipt = res.receipt
    assert tuple(receipt) == RECEIPT_KEYS
    assert receipt["actor"] == "hephaestus:T1"
    assert receipt["task_id"] == "T1"
    assert receipt["llm"] == {"calls": 2, "input_tokens": 4300, "output_tokens": 300}
    assert len(receipt["tools"]) == 1
    assert receipt["tools"][0]["tool"] == "read_file"
    assert receipt["tools"][0]["status"] == "ok"
    assert isinstance(receipt["tools"][0]["duration_ms"], int)
    assert receipt["one_line"].startswith("hephaestus:T1 · 2 calls · 4.3k in / 300 out")
    assert "read_file" in receipt["one_line"]

    leaked = json.dumps(receipt)
    for secret in (
        "SYSTEM_SECRET_TOKEN",
        "USER_SECRET_TOKEN",
        "SECRET_ASSISTANT_TEXT",
        "SECRET_FINAL_ANSWER",
        "hello",
        str(ws),
    ):
        assert secret not in leaked

    stored = [event for event in store.list_events(run_id) if event.event_type == "turn.receipt"]
    assert len(stored) == 1
    assert stored[0].stage == "orchestrate"
    assert set(stored[0].payload) == set(RECEIPT_KEYS)
    assert stored[0].payload["one_line"] == receipt["one_line"]
    assert captured[-1] == receipt


def test_loop_emits_receipt_when_budget_exhausted(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state.db")
    run_id = store.create_run(title="Budget", budget_policy=BudgetPolicy(max_llm_calls=1))
    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="calling",
                tool_calls=(
                    ToolCall(id="t1", name="read_file", arguments={"path": "/missing"}),
                ),
            )
        ]
    )
    loop = AgentLoop(provider=provider, store=store, run_id=run_id, max_steps=3)
    res = loop.run(system_prompt="SYS", user_prompt="GO")

    assert res.completed is False
    assert res.stop_reason.startswith("budget_exhausted")
    assert tuple(res.receipt) == RECEIPT_KEYS
    assert res.receipt["llm"]["calls"] == 1
    assert any(event.event_type == "turn.receipt" for event in store.list_events(run_id))


def test_write_manifest_md_captures_outcome_and_diff(tmp_path: Path) -> None:
    task = TaskNode(
        task_id="T1",
        title="Implement waitlist",
        objective="Add waitlist.py with a bounded queue",
        write_scope=["waitlist.py"],
        acceptance_checks=[
            AcceptanceCheck(
                id="C1",
                kind=CheckKind.COMMAND,
                description="Run tests",
                command="python3 -c 'pass'",
            )
        ],
        status=TaskStatus.VERIFYING,
        attempt=2,
        review_cycle=1,
    )
    checks = [
        AcceptanceCheck(
            id="C1",
            kind=CheckKind.COMMAND,
            description="Run tests",
            command="python3 -c 'pass'",
            passed=False,
        )
    ]
    verdict = ReviewVerdict(
        task_id="T1",
        attempt=2,
        passed=False,
        summary="Missing overflow guard",
    )
    receipt = build_turn_receipt(
        turn_id="turn-abc",
        actor="hephaestus:T1",
        task_id="T1",
        events=[{"kind": "llm", "input_tokens": 1000, "output_tokens": 50}],
    )
    diff_text = (
        "--- /dev/null\n"
        "+++ b/waitlist.py\n"
        "import queue\n"
        "QUEUE = queue.Queue\n"
    )
    art_dir = tmp_path / "attempt-2"
    art_dir.mkdir()

    manifest_path = _write_manifest_md(
        art_dir=art_dir,
        task=task,
        diff_text=diff_text,
        checks=checks,
        verdict=verdict,
        receipt=receipt,
    )

    assert manifest_path == art_dir / "MANIFEST.md"
    text = manifest_path.read_text(encoding="utf-8")
    assert "# MANIFEST — T1 (attempt 2)" in text
    assert "Outcome: rejected" in text
    assert "Checks: 0/1 passed" in text
    assert "Diff: +2/-0 lines" in text
    assert "Missing overflow guard" in text
    assert "hephaestus:T1 · 1 calls · 1.0k in / 50 out" in text
    assert "C1 (command): not passed — Run tests" in text
