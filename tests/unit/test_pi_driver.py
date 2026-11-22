"""Offline PiDriver protocol tests and heterogeneous Stage 2 integration tests.

Every test drives a deterministic local fake ``pi`` Python script: zero network,
zero Node.js, and zero real LLM calls.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetPolicy,
    GoalSpec,
    RunMode,
    RunStatus,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.harness.activity import milestone_fields
from olympus.harness.driver import (
    ActiveTurnHandle,
    DriverTurnRequest,
    ExecutionDriversConfig,
    TurnOutcome,
)
from olympus.harness.pi import PiDriver, PiNotAvailableError
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
from olympus.stages.align import AlignStage
from olympus.stages.orchestrate import OrchestrateStage

_FAKE_PI_SOURCE = '''
"""Deterministic fake Pi CLI for Olympus tests: offline, no Node, no network."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

RPC_SUPPORTED = False


def _write_json(name, payload):
    path = os.environ.get(name)
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)


def _emit(record):
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\\n")
    sys.stdout.flush()


def _usage(tin, tout):
    cache_read = int(os.environ.get("FAKE_PI_CACHE_READ", "0"))
    cache_write = int(os.environ.get("FAKE_PI_CACHE_WRITE", "0"))
    return {
        "input": tin,
        "output": tout,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "totalTokens": tin + tout + cache_read + cache_write,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def _assistant(text, *, stop, tin, tout, tool=None):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if tool is not None:
        content.append(
            {
                "type": "toolCall",
                "id": tool["id"],
                "name": tool["name"],
                "arguments": tool["arguments"],
            }
        )
    return {
        "role": "assistant",
        "content": content,
        "provider": "fake",
        "model": "fake-pi-model",
        "usage": _usage(tin, tout),
        "stopReason": stop,
        "timestamp": int(time.time() * 1000),
    }


def _write_target():
    relative = os.environ.get("FAKE_PI_WRITE_FILE", "feature.py")
    target = os.path.join(os.getcwd(), relative)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(os.environ.get("FAKE_PI_WRITE_CONTENT", "FLAG = True\\n"))
    return relative


def _emit_work():
    _emit({"type": "agent_start"})
    _emit({"type": "turn_start"})
    _emit({"type": "message_start", "message": {"role": "assistant", "content": []}})
    _emit(
        {
            "type": "message_update",
            "usage": _usage(0, 0),
            "assistantMessageEvent": {
                "type": "text_delta",
                "contentIndex": 0,
                "delta": "Working on it ",
            },
        }
    )
    target = _write_target()
    first = _assistant(
        "Working on it",
        stop="toolUse",
        tin=11,
        tout=5,
        tool={
            "id": "call-1",
            "name": "write",
            "arguments": {"path": target, "content": "FLAG = True\\n"},
        },
    )
    _emit({"type": "message_end", "message": first})
    _emit(
        {
            "type": "tool_execution_start",
            "toolCallId": "call-1",
            "toolName": "write",
            "args": {"path": target},
        }
    )
    _emit(
        {
            "type": "tool_execution_end",
            "toolCallId": "call-1",
            "toolName": "write",
            "result": {"content": [{"type": "text", "text": "wrote " + target}]},
            "isError": False,
            "durationMs": 12,
        }
    )
    if os.environ.get("FAKE_PI_WATCHDOG_ERRORS") == "1":
        for step in range(3):
            _emit({
                "type": "tool_execution_end",
                "toolCallId": f"err-{step}",
                "toolName": "read",
                "result": {"content": [{"type": "text", "text": "Repeated error"}]},
                "isError": True,
            })
    if os.environ.get("FAKE_PI_COMPACT") == "1":
        _emit({"type": "compaction_start", "reason": "threshold"})
        _emit({
            "type": "compaction_end",
            "reason": "threshold",
            "result": {"summary": "SECRET COMPACTION PROSE"},
            "aborted": False,
        })
    _emit({"type": "turn_end", "message": first, "toolResults": []})
    _emit({"type": "agent_end", "messages": [first], "willRetry": False})


def _finish(text):
    final = _assistant(text, stop="stop", tin=7, tout=3)
    _emit({"type": "message_start", "message": {"role": "assistant", "content": []}})
    _emit(
        {
            "type": "message_update",
            "usage": _usage(7, 3),
            "assistantMessageEvent": {
                "type": "text_delta",
                "contentIndex": 0,
                "delta": text,
            },
        }
    )
    _emit({"type": "message_end", "message": final})
    _emit({"type": "turn_end", "message": final, "toolResults": []})
    _emit({"type": "agent_end", "messages": [final], "willRetry": False})
    _emit({"type": "agent_settled", "aborted": False})


def _respond(command_id, command, data):
    record = {"type": "response", "command": command, "success": True}
    if command_id:
        record["id"] = command_id
    if data is not None:
        record["data"] = data
    _emit(record)


def _prepare_hang():
    if os.environ.get("FAKE_PI_HANG") != "1":
        return False
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    _write_json(
        "FAKE_PI_PID_FILE",
        {"pid": os.getpid(), "pgid": os.getpgid(0), "child": child.pid},
    )
    return True


def rpc_main():
    hang = _prepare_hang()
    await_steer = os.environ.get("FAKE_PI_AWAIT_STEER") == "1"
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = command.get("type")
        if kind == "prompt":
            _respond(command.get("id"), "prompt", {"disposition": "started"})
            _emit_work()
            if hang:
                break
            if not await_steer:
                _finish("Documented change.")
                return 0
        elif kind == "steer":
            _respond(command.get("id"), "steer", {"disposition": "queued"})
            _emit(
                {
                    "type": "queue_update",
                    "steering": [command.get("message", "")],
                    "followUp": [],
                }
            )
            _finish("steered:" + str(command.get("message", "")))
            return 0
        elif kind == "abort":
            _respond(command.get("id"), "abort", None)
            _emit({"type": "agent_settled", "aborted": True})
            return 0
    if hang:
        while True:
            time.sleep(0.1)
    return 0


def json_main():
    _emit(
        {
            "type": "session",
            "version": 3,
            "id": "fake-session",
            "timestamp": "2024-12-03T14:00:00.000Z",
            "cwd": os.getcwd(),
        }
    )
    _emit_work()
    _finish("Documented change.")
    return 0


def main():
    argv = sys.argv[1:]
    _write_json("FAKE_PI_ARGV_FILE", {"argv": argv})
    _write_json(
        "FAKE_PI_ENV_FILE",
        {
            "write_scope": os.environ.get("OLYMPUS_PI_WRITE_SCOPE"),
            "forbidden_globs": os.environ.get("OLYMPUS_PI_FORBIDDEN_GLOBS"),
            "cwd": os.getcwd(),
        },
    )
    if "--help" in argv or "-h" in argv:
        print("fake pi - deterministic Olympus test harness")
        if RPC_SUPPORTED:
            print("--mode <mode>                  Output mode: text (default), json, or rpc")
        else:
            print("--mode <mode>                  Output mode: text (default) or json")
        return 0
    if RPC_SUPPORTED and "--mode" in argv and argv[argv.index("--mode") + 1] == "rpc":
        return rpc_main()
    return json_main()


if __name__ == "__main__":
    sys.exit(main())
'''


def _write_fake_pi(root: Path, *, rpc: bool) -> Path:
    path = root / "fake_pi.py"
    source = _FAKE_PI_SOURCE.replace("RPC_SUPPORTED = False", f"RPC_SUPPORTED = {rpc!r}")
    path.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _request(workspace: Path, *, read_only: bool = False) -> DriverTurnRequest:
    return DriverTurnRequest(
        workspace_path=workspace,
        env_root=workspace,
        system_prompt="You are the fake worker.",
        user_prompt="Implement the flag module.",
        actor="hephaestus:T1",
        task_id="T1",
        read_only=read_only,
        write_scope=[] if read_only else ["feature.py"],
        forbidden_globs=[] if read_only else ["shared/*.py"],
        max_steps=5,
        trace_path=workspace / "worker.jsonl",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _process_alive(pid: int) -> bool:
    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    try:
        state = content.rsplit(")", 1)[1].split()[0]
    except IndexError:
        return False
    return state != "Z"


def _wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not met before the timeout")


def _process_group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def _wait_for_pids(path: Path) -> dict[str, int]:
    _wait_for(path.is_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"pid": int(data["pid"]), "pgid": int(data["pgid"]), "child": int(data["child"])}




def test_pi_driver_streams_rpc_events_usage_receipt_and_steer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PI_AWAIT_STEER", "1")
    fake = _write_fake_pi(tmp_path, rpc=True)
    argv_file = tmp_path / "argv.json"
    env_file = tmp_path / "env.json"
    monkeypatch.setenv("FAKE_PI_ARGV_FILE", str(argv_file))
    monkeypatch.setenv("FAKE_PI_ENV_FILE", str(env_file))
    store = RunStore(tmp_path / "runs.db")
    run_id = store.create_run(title="pi rpc turn")
    trace = tmp_path / "worker.jsonl"
    driver = PiDriver(binary=str(fake), store=store, run_id=run_id)

    handles: list[ActiveTurnHandle] = []
    supports_at_start: list[bool] = []
    events: list[tuple[str, dict[str, Any]]] = []
    steered: list[bool] = []

    def on_event(kind: str, payload: dict[str, Any]) -> None:
        events.append((kind, payload))
        if kind == "agent.tool_finished" and not steered:
            steered.append(handles[0].steer("Prefer smaller edits", source="human"))

    def on_handle(handle: ActiveTurnHandle) -> None:
        handles.append(handle)
        supports_at_start.append(handle.supports_steer)

    outcome = driver.run_turn(
        _request(tmp_path), on_event=on_event, on_handle=on_handle
    )

    assert supports_at_start == [True]
    assert steered == [True]
    assert outcome.completed and outcome.stop_reason == "final_response"
    assert outcome.driver_name == "pi"
    assert outcome.final_content == "steered:Prefer smaller edits"
    assert outcome.steps_taken == 2
    assert (outcome.total_input_tokens, outcome.total_output_tokens) == (18, 8)
    assert outcome.receipt["llm"] == {"calls": 2, "input_tokens": 18, "output_tokens": 8}
    assert outcome.receipt["tools"] == [{"tool": "write", "status": "ok", "duration_ms": 12}]
    assert "2 calls" in outcome.receipt["one_line"]
    assert "Prefer smaller edits" not in json.dumps(outcome.receipt)

    kinds = [kind for kind, _ in events]
    assert (
        kinds.index("agent.tool_started")
        < kinds.index("agent.tool_finished")
        < kinds.index("agent.steered")
    )
    steered_payloads = [payload for kind, payload in events if kind == "agent.steered"]
    assert steered_payloads == [
        {"actor": "hephaestus:T1", "task_id": "T1", "step": 2, "source": "human"}
    ]
    assert "Prefer smaller edits" not in json.dumps(steered_payloads)

    ledger = store.get_run(run_id)["budget_ledger"]
    assert ledger.llm_calls_used == 2
    assert ledger.llm_calls_reserved == 0
    assert (ledger.input_tokens, ledger.output_tokens) == (18, 8)
    receipts = [e for e in store.list_events(run_id) if e.event_type == "turn.receipt"]
    assert len(receipts) == 1 and receipts[0].payload == outcome.receipt

    entries = _read_jsonl(trace)
    normalized = [entry for entry in entries if "event" in entry]
    assert any(entry["event"] == "tool_call" and entry["tool"] == "write" for entry in normalized)
    assert any(
        entry["event"] == "tool_end" and entry["duration_ms"] == 12 for entry in normalized
    )
    assert any(
        entry["event"] == "agent.steered" and entry["source"] == "human"
        for entry in normalized
    )
    assert not any(
        entry.get("record", {}).get("type") == "message_update" for entry in entries
    )
    text_deltas = [payload for kind, payload in events if kind == "text_delta"]
    assert any(payload["text"] == "Working on it " for payload in text_deltas)
    assert any(
        entry["event"] == "llm" and entry["input_tokens"] == 11 for entry in normalized
    )

    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert argv[:2] == ["--mode", "rpc"]
    assert "--no-session" in argv and "--no-extensions" in argv
    assert "--no-context-files" in argv and "--no-skills" in argv
    assert "--extension" in argv and argv[argv.index("--extension") + 1].endswith(
        "pi_scope_guard.ts"
    )
    assert "-p" not in argv
    env = json.loads(env_file.read_text(encoding="utf-8"))
    assert json.loads(env["write_scope"]) == ["feature.py"]
    assert json.loads(env["forbidden_globs"]) == ["shared/*.py"]
    assert env["cwd"] == str(tmp_path)


def test_pi_driver_settles_cache_tokens_and_enforces_true_total_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PI_CACHE_READ", "50")
    monkeypatch.setenv("FAKE_PI_CACHE_WRITE", "10")
    store = RunStore(tmp_path / "run.db")
    run_id = store.create_run(
        title="cache-token-cap",
        mode=RunMode.SIMULATION,
        budget_policy=BudgetPolicy(max_total_tokens=50),
    )
    fake = _write_fake_pi(tmp_path, rpc=True)
    trace_path = tmp_path / "trace.jsonl"
    events: list[tuple[str, dict[str, Any]]] = []

    outcome = PiDriver(binary=str(fake), store=store, run_id=run_id).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="",
            user_prompt="test",
            trace_path=trace_path,
        ),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    ledger = store.get_run(run_id)["budget_ledger"]
    assert not outcome.completed
    assert outcome.stop_reason.startswith("budget_exhausted")
    assert outcome.steps_taken == ledger.llm_calls_used == 2
    assert (ledger.input_tokens, ledger.output_tokens) == (18, 8)
    assert (ledger.cache_read_tokens, ledger.cache_write_tokens) == (100, 20)
    # The same run stays under the cap without cache tokens; cached reads/writes trip it.
    assert ledger.total_tokens == 146 > 50

    llm_events = [payload for kind, payload in events if kind == "llm"]
    assert all(
        (payload["cache_read_tokens"], payload["cache_write_tokens"]) == (50, 10)
        for payload in llm_events
    )
    llm_trace = [entry for entry in _read_jsonl(trace_path) if entry.get("event") == "llm"]
    assert llm_trace and all(
        (entry["cache_read_tokens"], entry["cache_write_tokens"]) == (50, 10)
        for entry in llm_trace
    )


def test_pi_driver_bridges_compaction_milestones_without_raw_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PI_COMPACT", "1")
    fake = _write_fake_pi(tmp_path, rpc=True)
    trace_path = tmp_path / "trace.jsonl"
    events: list[tuple[str, dict[str, Any]]] = []

    outcome = PiDriver(binary=str(fake)).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="",
            user_prompt="test",
            trace_path=trace_path,
        ),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    assert outcome.completed
    compactions = [payload for kind, payload in events if kind == "agent.compaction"]
    assert [payload["phase"] for payload in compactions] == ["start", "end"]
    assert "SECRET COMPACTION PROSE" not in json.dumps(compactions)

    trace_compactions = [
        entry for entry in _read_jsonl(trace_path) if entry.get("event") == "agent.compaction"
    ]
    assert [entry["phase"] for entry in trace_compactions] == ["start", "end"]


def test_compaction_milestone_fields_drop_raw_prose() -> None:
    payload = {"phase": "end", "result": {"summary": "SECRET COMPACTION PROSE"}}
    assert milestone_fields("agent.compaction", payload) == {"phase": "end"}
    assert milestone_fields("agent.compaction", {"phase": "middle"}) is None


def test_pi_driver_cancel_reclaims_process_group_with_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _write_fake_pi(tmp_path, rpc=True)
    pid_file = tmp_path / "pids.json"
    monkeypatch.setenv("FAKE_PI_HANG", "1")
    monkeypatch.setenv("FAKE_PI_PID_FILE", str(pid_file))
    driver = PiDriver(
        binary=str(fake), cancel_grace_seconds=0.2, term_grace_seconds=0.2
    )

    handles: list[ActiveTurnHandle] = []
    outcomes: list[TurnOutcome] = []

    def run_turn() -> None:
        outcomes.append(driver.run_turn(_request(tmp_path), on_handle=handles.append))

    worker = Thread(target=run_turn)
    worker.start()
    pids = _wait_for_pids(pid_file)
    _wait_for(lambda: bool(handles))
    assert handles[0].supports_steer

    started = time.monotonic()
    handles[0].cancel(reason="human_paused")
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert time.monotonic() - started < 8

    assert outcomes[0].stop_reason == "cancelled" and not outcomes[0].completed
    _wait_for(lambda: not _process_alive(pids["pid"]))
    _wait_for(lambda: not _process_alive(pids["child"]))
    _wait_for(lambda: _process_group_gone(pids["pgid"]))


def test_pi_driver_falls_back_to_json_mode_without_rpc_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _write_fake_pi(tmp_path, rpc=False)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_PI_ARGV_FILE", str(argv_file))
    driver = PiDriver(binary=str(fake))
    handles: list[ActiveTurnHandle] = []
    events: list[str] = []

    outcome = driver.run_turn(
        _request(tmp_path),
        on_event=lambda kind, payload: events.append(kind),
        on_handle=handles.append,
    )

    assert outcome.completed and outcome.final_content == "Documented change."
    assert outcome.receipt["llm"] == {"calls": 2, "input_tokens": 18, "output_tokens": 8}
    assert "agent.tool_started" in events and "agent.tool_finished" in events
    assert not handles[0].supports_steer
    assert handles[0].steer("too late", source="human") is False

    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert argv[:3] == ["-p", "--mode", "json"]
    assert "--" in argv and argv[-1] == "Implement the flag module."
    assert "rpc" not in argv


def test_pi_driver_missing_binary_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(PiNotAvailableError) as excinfo:
        PiDriver(binary=str(tmp_path / "missing-pi"))
    message = str(excinfo.value)
    assert "not found" in message and "OLYMPUS_PI_BIN" in message

    fake = _write_fake_pi(tmp_path, rpc=True)
    driver = PiDriver(binary=str(fake))
    fake.unlink()
    with pytest.raises(PiNotAvailableError):
        driver.run_turn(_request(tmp_path))


def test_pi_driver_read_only_turn_restricts_pi_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _write_fake_pi(tmp_path, rpc=True)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_PI_ARGV_FILE", str(argv_file))
    outcome = PiDriver(binary=str(fake)).run_turn(_request(tmp_path, read_only=True))

    assert outcome.completed
    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert argv[argv.index("--tools") + 1] == "read,grep,find,ls"
    assert argv[argv.index("--exclude-tools") + 1] == "bash,write,edit,powershell"
    assert "--no-extensions" in argv
    assert "--extension" not in argv




def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _confirmed_run(root: Path) -> tuple[RunStore, str]:
    store = RunStore(root / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Pi heterogeneous run", mode=RunMode.SIMULATION)
    spec = GoalSpec(
        spec_id="spec-pi",
        goal="Add a feature flag module",
        requirements=[
            {
                "id": "R1",
                "description": "Provide feature.py with FLAG = True",
                "acceptance_criteria": ["feature.FLAG is True"],
            }
        ],
    )
    store.save_spec(run_id, spec)
    AlignStage(
        store=store, provider=MockLLMProvider(), workspace_root=root
    ).confirm_spec(run_id=run_id, approver="zeus")
    return store, run_id


def _single_task_plan(
    spec: GoalSpec, checks: list[dict[str, Any]], *, max_cycles: int = 3
) -> TaskPlan:
    return TaskPlan.model_validate(
        {
            "plan_id": "plan-pi-e2e",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "tasks": [
                {
                    "task_id": "T1",
                    "title": "Implement feature.py",
                    "objective": "Implement the assigned flag module",
                    "requirement_refs": ["R1"],
                    "write_scope": ["feature.py"],
                    "max_cycles": max_cycles,
                    "acceptance_checks": checks,
                }
            ],
        }
    )


def _approve_plan(store: RunStore, run_id: str, plan: TaskPlan) -> None:
    store.save_plan(run_id, plan, actor="orchestrate")
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )


def _review_responder(
    messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
) -> LLMResponse:
    del tools
    assert "Independent Verifier" in messages[0].content
    return LLMResponse(
        content=json.dumps(
            {
                "task_id": "T1",
                "attempt": 1,
                "passed": True,
                "summary": "Scope-clean implementation verified.",
                "findings": [],
            }
        )
    )


@pytest.mark.parametrize("watchdog_enabled", [False, True])
def test_pi_worker_heterogeneous_run_with_argus_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, watchdog_enabled: bool
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec,
        [
            {
                "id": "CHK-1",
                "kind": "command",
                "description": "Import the flag module",
                "command": "python3 -c 'import feature; assert feature.FLAG'",
            },
            {
                "id": "CHK-2",
                "kind": "independent_review",
                "description": "Cold-start independent review",
            },
        ],
    )
    _approve_plan(store, run_id, plan)
    fake = _write_fake_pi(tmp_path, rpc=True)
    argv_file = tmp_path / "argv.json"
    env_file = tmp_path / "env.json"
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(fake))
    monkeypatch.setenv("FAKE_PI_ARGV_FILE", str(argv_file))
    monkeypatch.setenv("FAKE_PI_ENV_FILE", str(env_file))

    monitor = None
    if watchdog_enabled:
        monkeypatch.setenv("FAKE_PI_AWAIT_STEER", "1")
        monkeypatch.setenv("FAKE_PI_WATCHDOG_ERRORS", "1")
        monitor = MockLLMProvider(responses=[LLMResponse(content=(
            '{"verdict":"STUCK_LOOP","summary":"Repeated read errors.",'
            '"suggested_steer":"Continue without the blocked read."}'
        ))])
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_review_responder),
        workspace_root=tmp_path,
        monitor_provider=monitor,
        execution_drivers=ExecutionDriversConfig(
            worker_driver="pi", reviewer_driver="native"
        ),
    )
    snap = stage.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED

    events = store.list_events(run_id)
    if watchdog_enabled:
        assert monitor is not None and len(monitor.recorded_requests) == 1
        assert any(e.event_type == "watchdog.checked" for e in events)
        assert any(
            e.event_type == "agent.steered" and e.payload["source"] == "watchdog"
            for e in events
        )
    verdicts = [e for e in events if e.event_type == "review.verdict_recorded"]
    assert len(verdicts) == 1 and verdicts[0].payload["passed"] is True
    assert not [e for e in events if e.event_type == "task.diff_gate_rejected"]

    ledger = store.get_run(run_id)["budget_ledger"]
    assert (ledger.input_tokens, ledger.output_tokens) == (18, 8)
    watchdog_calls = 1 if watchdog_enabled else 0
    assert ledger.llm_calls_used == 3 + watchdog_calls

    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert "--mode" in argv and argv[argv.index("--mode") + 1] == "rpc"
    assert "--extension" in argv and argv[argv.index("--extension") + 1].endswith(
        "pi_scope_guard.ts"
    )
    env = json.loads(env_file.read_text(encoding="utf-8"))
    assert json.loads(env["write_scope"]) == ["feature.py"]

    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    tool_entries = [
        entry for entry in _read_jsonl(attempt_dir / "worker.jsonl") if entry.get("event")
    ]
    assert any(entry["event"] == "tool_end" and entry["tool"] == "write" for entry in tool_entries)
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "feature.py").is_file()
    assert not (tmp_path / "feature.py").exists()


def test_pi_worker_out_of_scope_edit_rejected_by_diff_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec,
        [
            {
                "id": "CHK-1",
                "kind": "command",
                "description": "Write a marker proving checks never ran",
                "command": (
                    "python3 -c 'from pathlib import Path; "
                    'Path("check-ran.txt").write_text("ran")\''
                ),
            }
        ],
        max_cycles=1,
    )
    _approve_plan(store, run_id, plan)
    fake = _write_fake_pi(tmp_path, rpc=True)
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(fake))
    monkeypatch.setenv("FAKE_PI_WRITE_FILE", "docs/secret.md")
    monkeypatch.setenv("FAKE_PI_WRITE_CONTENT", "leaked\n")

    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="pi"),
    )
    snap = stage.run_execution(run_id=run_id)

    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.last_error is not None
    assert "docs/secret.md" in task.last_error
    assert "outside authorized write_scope" in task.last_error

    events = store.list_events(run_id)
    rejections = [e for e in events if e.event_type == "task.diff_gate_rejected"]
    assert len(rejections) == 1
    assert rejections[0].payload["violating_paths"] == ["docs/secret.md"]
    assert rejections[0].payload["allowed_write_scope"] == ["feature.py"]
    assert not [e for e in events if e.event_type == "review.verdict_recorded"]
    assert not [e for e in events if e.event_type == "check.command_completed"]

    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert (attempt_dir / "diff.patch").is_file()
    assert not list(attempt_dir.glob("check-*.log"))
    assert not list(attempt_dir.glob("verdict-cycle-*.json"))
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert not (overlay / "docs" / "secret.md").exists()


def test_missing_pi_binary_fails_before_task_lease_or_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec,
        [
            {
                "id": "CHK-1",
                "kind": "command",
                "description": "Import the flag module",
                "command": "python3 -c 'import feature; assert feature.FLAG'",
            }
        ],
    )
    _approve_plan(store, run_id, plan)
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(tmp_path / "missing-pi"))
    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="pi"),
    )

    with pytest.raises(PiNotAvailableError) as excinfo:
        stage.run_execution(run_id=run_id)

    message = str(excinfo.value)
    assert "not found" in message and "OLYMPUS_PI_BIN" in message
    snap = store.get_run(run_id)
    assert snap["plan"].tasks[0].status == TaskStatus.WAITING
    assert not [
        e
        for e in store.list_events(run_id)
        if e.event_type in ("task.verifying", "review.verdict_recorded", "task.diff_gate_rejected")
    ]


def _proc_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_pi_driver_cancel_reclaims_process_group_under_stdin_backpressure(tmp_path: Path) -> None:
    fake = tmp_path / "pi-backpressure"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        "if '--help' in sys.argv:\n"
        "    print('--mode json rpc')\n"
        "else:\n"
        "    print('pid', os.getpid(), flush=True)\n"
        "    time.sleep(20)\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    ready, finished, cancelled = Event(), Event(), Event()
    handles: list[ActiveTurnHandle] = []
    driver = PiDriver(binary=str(fake), cancel_grace_seconds=0.1, term_grace_seconds=0.1)
    trace_path = tmp_path / "trace.jsonl"
    req = DriverTurnRequest(
        workspace_path=tmp_path,
        env_root=tmp_path,
        system_prompt="",
        user_prompt="x" * 1_048_576,
        actor="worker",
        trace_path=trace_path,
    )

    def register(handle: ActiveTurnHandle) -> None:
        handles.append(handle)
        ready.set()

    worker = Thread(
        target=lambda: (driver.run_turn(req, on_handle=register), finished.set()),
        daemon=True,
    )
    worker.start()
    assert ready.wait(3.0)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and (
        not trace_path.exists() or "pi.unparsed" not in trace_path.read_text(encoding="utf-8")
    ):
        time.sleep(0.01)
    pid = next(
        int(entry["line"].split()[1])
        for entry in (
            json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
        )
        if entry.get("source") == "pi.unparsed"
    )
    stopper = Thread(
        target=lambda: (handles[0].cancel(reason="watchdog_paused"), cancelled.set()),
        daemon=True,
    )
    stopper.start()
    try:
        assert cancelled.wait(2.0)
        assert finished.wait(2.0)
        assert not _proc_alive(pid)
    finally:
        if _proc_alive(pid):
            os.killpg(pid, signal.SIGKILL)
        worker.join(2.0)


def test_pi_driver_help_probe_reaps_helper_child_process(tmp_path: Path) -> None:
    fake = tmp_path / "pi-help-helper"
    msg_end = json.dumps({"type": "message_end", "message": {"role": "assistant", "content": "ok"}})
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, subprocess, sys\n"
        "if '--help' in sys.argv:\n"
        "    child = subprocess.Popen(['sleep', '20'], stdin=subprocess.DEVNULL,\n"
        "        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "    pathlib.Path(__file__).with_name('helper.pid').write_text(str(child.pid))\n"
        "    print('--mode json rpc')\n"
        "else:\n"
        "    sys.stdin.readline()\n"
        f"    print({msg_end!r})\n"
        "    print('{\"type\":\"agent_settled\"}', flush=True)\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    req = DriverTurnRequest(
        workspace_path=tmp_path,
        env_root=tmp_path,
        system_prompt="",
        user_prompt="test",
        actor="worker",
    )
    outcome = PiDriver(binary=str(fake)).run_turn(req)
    helper_pid = int((tmp_path / "helper.pid").read_text(encoding="utf-8"))
    time.sleep(0.05)
    try:
        assert outcome.completed
        assert not _proc_alive(helper_pid)
    finally:
        if _proc_alive(helper_pid):
            os.kill(helper_pid, signal.SIGKILL)


def test_pi_driver_enforces_budget_and_max_steps_with_honest_ledger(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run.db")
    run_id = store.create_run(
        title="budget-cap",
        mode=RunMode.SIMULATION,
        budget_policy=BudgetPolicy(max_llm_calls=1),
    )
    fake = _write_fake_pi(tmp_path, rpc=True)
    trace_path = tmp_path / "trace.jsonl"
    outcome = PiDriver(binary=str(fake), store=store, run_id=run_id).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="",
            user_prompt="test",
            max_steps=1,
            trace_path=trace_path,
        )
    )
    ledger = store.get_run(run_id)["budget_ledger"]
    assert not outcome.completed
    assert outcome.stop_reason.startswith("budget_exhausted")
    assert ledger.llm_calls_used == outcome.steps_taken
    assert ledger.input_tokens == outcome.total_input_tokens
    assert ledger.output_tokens == outcome.total_output_tokens
    assert not any(
        "usage.settle_error" in line for line in trace_path.read_text(encoding="utf-8").splitlines()
    )


def test_pi_driver_trace_omits_message_start_turn_end_and_success_responses(
    tmp_path: Path,
) -> None:
    fake = _write_fake_pi(tmp_path, rpc=True)
    trace_path = tmp_path / "compact.jsonl"
    deltas: list[str] = []
    outcome = PiDriver(binary=str(fake)).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="",
            user_prompt="test",
            trace_path=trace_path,
        ),
        on_event=lambda kind, payload: (
            deltas.append(str(payload.get("text", ""))) if kind == "text_delta" else None
        ),
    )
    assert outcome.completed
    assert deltas
    raw_records = [
        entry["record"]
        for entry in _read_jsonl(trace_path)
        if entry.get("source") == "pi" and isinstance(entry.get("record"), dict)
    ]
    types = {record.get("type") for record in raw_records}
    assert "message_update" not in types
    assert "message_start" not in types
    assert "turn_end" not in types
    assert not any(
        record.get("type") == "response" and record.get("success") is True
        for record in raw_records
    )
    assert "message_end" in types

