"""Offline PiDriver protocol tests and heterogeneous Stage 2 integration tests.

Every test drives a deterministic local fake ``pi`` Python script: zero network,
zero Node.js, and zero real LLM calls.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event, Lock, Thread
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
    SpecStatus,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.harness.activity import milestone_fields
from olympus.harness.driver import (
    ActiveTurnHandle,
    DriverRegistry,
    DriverTurnRequest,
    ExecutionDriversConfig,
    TurnOutcome,
    resolve_driver_spec,
)
from olympus.harness.pi import (
    _SCOPE_GUARD_PATH,
    PiDriver,
    PiNotAvailableError,
    _reset_pi_availability_cache,
    _safe_argv,
)
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
from olympus.stages.align import AlignStage
from olympus.stages.design import DesignStage
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
    reasoning = int(os.environ.get("FAKE_PI_REASONING", "0"))
    return {
        "input": tin,
        "output": tout,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "reasoning": reasoning,
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


def _final_text():
    payload_path = os.environ.get("FAKE_PI_PAYLOAD_FILE")
    if payload_path:
        with open(payload_path, encoding="utf-8") as handle:
            return handle.read()
    return "Documented change."


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
                _finish(_final_text())
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
    _finish(_final_text())
    return 0


def main():
    argv = sys.argv[1:]
    _write_json("FAKE_PI_ARGV_FILE", {"argv": argv})
    _write_json(
        "FAKE_PI_ENV_FILE",
        {
            "write_scope": os.environ.get("OLYMPUS_PI_WRITE_SCOPE"),
            "forbidden_globs": os.environ.get("OLYMPUS_PI_FORBIDDEN_GLOBS"),
            "strict_scope": os.environ.get("OLYMPUS_PI_STRICT_SCOPE"),
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
    assert outcome.receipt["llm"] == {
        "calls": 2,
        "input_tokens": 18,
        "output_tokens": 8,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 26,
    }
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
    assert outcome.receipt["llm"] == {
        "calls": 2,
        "input_tokens": 18,
        "output_tokens": 8,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 26,
    }
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


def test_pi_driver_writable_turn_appends_prompt_and_enables_native_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _write_fake_pi(tmp_path, rpc=True)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_PI_ARGV_FILE", str(argv_file))

    outcome = PiDriver(binary=str(fake)).run_turn(_request(tmp_path))

    assert outcome.completed
    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert "--append-system-prompt" in argv
    assert "--system-prompt" not in argv
    assert argv[argv.index("--append-system-prompt") + 1] == "You are the fake worker."
    assert argv[argv.index("--tools") + 1] == "read,bash,edit,write,grep,find,ls"
    assert "--exclude-tools" not in argv
    assert argv[argv.index("--extension") + 1].endswith("pi_scope_guard.ts")

    spawn = next(
        entry
        for entry in _read_jsonl(tmp_path / "worker.jsonl")
        if entry.get("event") == "pi.spawn"
    )
    assert f"<{len('You are the fake worker.')} chars>" in spawn["argv"]
    assert "You are the fake worker." not in json.dumps(spawn)


def test_safe_argv_redacts_prompt_flags_and_trailing_prompt() -> None:
    argv = _safe_argv(
        [
            "pi",
            "--append-system-prompt",
            "system rules",
            "--model",
            "fake-model",
            "--",
            "user prompt",
        ]
    )
    assert argv == [
        "--append-system-prompt",
        f"<{len('system rules')} chars>",
        "--model",
        "fake-model",
        "--",
        f"<prompt: {len('user prompt')} chars>",
    ]


def test_pi_driver_read_only_turn_restricts_pi_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _write_fake_pi(tmp_path, rpc=True)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_PI_ARGV_FILE", str(argv_file))
    outcome = PiDriver(binary=str(fake)).run_turn(_request(tmp_path, read_only=True))

    assert outcome.completed
    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert "--append-system-prompt" in argv and "--system-prompt" not in argv
    assert argv[argv.index("--tools") + 1] == "read,grep,find,ls"
    assert argv[argv.index("--exclude-tools") + 1] == "bash,write,edit,powershell"
    assert "read,bash,edit,write,grep,find,ls" not in argv
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


def test_pi_worker_heterogeneous_run_with_argus_review(
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

    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_review_responder),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(
            worker_driver="pi", reviewer_driver="native"
        ),
    )
    snap = stage.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED

    events = store.list_events(run_id)
    verdicts = [e for e in events if e.event_type == "review.verdict_recorded"]
    assert len(verdicts) == 1 and verdicts[0].payload["passed"] is True
    assert not [e for e in events if e.event_type == "task.diff_gate_rejected"]

    ledger = store.get_run(run_id)["budget_ledger"]
    assert (ledger.input_tokens, ledger.output_tokens) == (18, 8)
    assert ledger.llm_calls_used == 3

    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert "--mode" in argv and argv[argv.index("--mode") + 1] == "rpc"
    assert "--extension" in argv and argv[argv.index("--extension") + 1].endswith(
        "pi_scope_guard.ts"
    )
    env = json.loads(env_file.read_text(encoding="utf-8"))
    assert json.loads(env["write_scope"]) == ["feature.py"]
    assert env["strict_scope"] == "0"

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
    env_file = tmp_path / "env.json"
    monkeypatch.setenv("FAKE_PI_ENV_FILE", str(env_file))

    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(),
        workspace_root=tmp_path,
        execution_drivers=ExecutionDriversConfig(worker_driver="pi"),
    )
    snap = stage.run_execution(run_id=run_id, strict_scope=True)

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
    env = json.loads(env_file.read_text(encoding="utf-8"))
    assert env["strict_scope"] == "1"

    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert (attempt_dir / "diff.patch").is_file()
    assert not list(attempt_dir.glob("check-*.log"))
    assert not list(attempt_dir.glob("verdict-cycle-*.json"))
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert not (overlay / "docs" / "secret.md").exists()


_GUARD_RUNNER_SOURCE = """
import { pathToFileURL } from "node:url";
process.chdir(process.env.OLYMPUS_GUARD_WORKSPACE);
const mod = await import(pathToFileURL(process.env.OLYMPUS_GUARD_PATH).href);
let handler = null;
mod.default({ on: (event, fn) => { if (event === "tool_call") handler = fn; } });
const results = [];
for (const [toolName, target] of JSON.parse(process.env.OLYMPUS_GUARD_CALLS)) {
  const result = await handler({ toolName, input: { path: target } });
  results.push(result && result.block ? { block: true, reason: result.reason } : { block: false });
}
console.log(JSON.stringify(results));
"""


def _run_scope_guard(
    tmp_path: Path,
    *,
    strict: bool,
    write_scope: Sequence[str],
    forbidden_globs: Sequence[str],
    calls: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    if shutil.which("node") is None:
        pytest.skip("node is required to execute pi_scope_guard.ts")
    workspace = tmp_path / "guard-ws"
    workspace.mkdir(parents=True)
    runner = tmp_path / "guard_runner.mjs"
    runner.write_text(_GUARD_RUNNER_SOURCE, encoding="utf-8")
    env = {
        **os.environ,
        "OLYMPUS_GUARD_PATH": str(_SCOPE_GUARD_PATH),
        "OLYMPUS_GUARD_WORKSPACE": str(workspace),
        "OLYMPUS_GUARD_CALLS": json.dumps([list(call) for call in calls]),
        "OLYMPUS_PI_STRICT_SCOPE": "1" if strict else "0",
        "OLYMPUS_PI_WRITE_SCOPE": json.dumps(list(write_scope)),
        "OLYMPUS_PI_FORBIDDEN_GLOBS": json.dumps(list(forbidden_globs)),
    }
    proc = subprocess.run(
        ["node", str(runner)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_pi_scope_guard_hard_blocks_escape_and_forbidden_and_relaxes_expected_scope(
    tmp_path: Path,
) -> None:
    relaxed = _run_scope_guard(
        tmp_path / "relaxed",
        strict=False,
        write_scope=["feature.py"],
        forbidden_globs=["contracts/", "sibling.py"],
        calls=[
            ("write", "feature.py"),
            ("write", "pkg/__init__.py"),
            ("edit", "docs/notes.md"),
            ("write", "contracts/api.py"),
            ("write", "sibling.py"),
            ("write", "../escape.py"),
        ],
    )
    assert [result["block"] for result in relaxed] == [False, False, False, True, True, True]
    assert "forbidden shared-contract glob" in relaxed[3]["reason"]
    assert "forbidden shared-contract glob" in relaxed[4]["reason"]
    assert "escapes the sandbox workspace" in relaxed[5]["reason"]

    strict = _run_scope_guard(
        tmp_path / "strict",
        strict=True,
        write_scope=["feature.py"],
        forbidden_globs=["contracts/"],
        calls=[("write", "feature.py"), ("write", "pkg/__init__.py")],
    )
    assert strict[0] == {"block": False}
    assert strict[1]["block"] is True
    assert "outside write_scope" in strict[1]["reason"]


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



# ------------------------------------------------------- stage driver auto-default seam


class _StageProvider:
    """Non-mock provider whose call count proves whether a stage used the native loop."""

    provider_name = "openai"
    supports_external_driver = True

    def __init__(self, model_name: str = "stage-model") -> None:
        self.model_name = model_name
        self.calls = 0

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
    ) -> LLMResponse:
        del messages, tools, temperature
        self.calls += 1
        raise AssertionError("the native loop must not run when auto resolves to Pi")


class _ScriptedStageDriver:
    """External driver returning pre-scripted structured payloads and token usage."""

    name = "scripted"

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.requests: list[DriverTurnRequest] = []
        self._lock = Lock()

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome:
        del on_event, on_handle
        with self._lock:
            self.requests.append(request)
            content = self.contents.pop(0)
        return TurnOutcome(
            completed=True,
            stop_reason="final_response",
            final_content=content,
            steps_taken=1,
            total_input_tokens=10,
            total_output_tokens=5,
            driver_name=self.name,
        )


def _stage_goal_payload() -> dict[str, Any]:
    return {
        "spec_id": "spec-stage-pi",
        "goal": "Expose a calculator module with an explicit zero-division error",
        "deliverable_kind": "library",
        "observable_surfaces": ["calculator module"],
        "actors": ["developer"],
        "test_seams": ["calculator.divide(a, b)"],
        "requirements": [
            {
                "id": "R1",
                "description": "divide returns a / b and rejects a zero divisor",
                "actor": "developer",
                "ears_criteria": [
                    {
                        "id": "R1-C1",
                        "pattern": "ubiquitous",
                        "outcome": "return a / b for a non-zero divisor",
                    },
                    {
                        "id": "R1-C2",
                        "pattern": "unwanted_behavior",
                        "condition": "b equals zero",
                        "outcome": "raise ValueError('zero')",
                    },
                ],
            }
        ],
        "out_of_scope": ["graphical interface"],
        "constraints": [],
    }


def _stage_design_payload() -> dict[str, Any]:
    return {
        "architecture_summary": "A single calculator module with an explicit zero error",
        "shared_invariants": [
            {
                "id": "INV-ERR",
                "category": "error_envelope",
                "rule": "Raise ValueError('zero') when the divisor is zero",
                "applies_to": ["M1"],
            }
        ],
        "shared_contract_paths": [],
        "modules": [
            {
                "id": "M1",
                "name": "Calculator",
                "responsibility": "Provide arithmetic helpers with explicit errors",
                "requirement_refs": ["R1"],
                "target_paths": ["calculator.py"],
            }
        ],
        "data_models": [],
        "interfaces": [],
        "execution_strategy": "serial_handoff",
        "strategy_rationale": "One cohesive module keeps the change atomic and verifiable.",
    }


def _stage_plan_payload(spec: GoalSpec) -> dict[str, Any]:
    return {
        "plan_id": "plan-stage-pi",
        "spec_id": spec.spec_id,
        "spec_hash": spec.content_hash,
        "tasks": [
            {
                "task_id": "T1",
                "title": "Implement calculator.divide",
                "objective": "Implement calculator.divide with a zero guard in calculator.py",
                "requirement_refs": ["R1"],
                "write_scope": ["calculator.py"],
                "acceptance_checks": [
                    {
                        "id": "C1",
                        "kind": "command",
                        "description": "Import the calculator module",
                        "command": "python3 -c 'import calculator'",
                    }
                ],
            }
        ],
    }


def _workspace_files(root: Path) -> set[str]:
    ignored = {".git", ".olympus"}
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not ignored.intersection(path.relative_to(root).parts)
    }


def _git_status_clean(root: Path) -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() == ""


def test_pi_driver_synthesizes_ephemeral_gateway_models_json_without_secret_literals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    capture_file = tmp_path / "pi-capture.json"
    fake_pi = tmp_path / "fake-pi"
    fake_pi.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if '--help' in sys.argv:\n"
        "    print('  --mode <mode>  Output mode: text, json, or rpc')\n"
        "    raise SystemExit(0)\n"
        "cfg_dir = os.environ.get('PI_CODING_AGENT_DIR', '')\n"
        "models_file = Path(cfg_dir, 'models.json') if cfg_dir else None\n"
        "models_json = models_file.read_text(encoding='utf-8') if models_file else ''\n"
        "payload = {'argv': sys.argv[1:], 'cfg_dir': cfg_dir, 'models_json': models_json}\n"
        f"Path({str(capture_file)!r}).write_text(json.dumps(payload), encoding='utf-8')\n"
        "msg = {\n"
        "    'type': 'message_end',\n"
        "    'message': {\n"
        "        'role': 'assistant',\n"
        "        'content': [{'type': 'text', 'text': 'ok'}],\n"
        "        'usage': {'input': 5, 'output': 2},\n"
        "    },\n"
        "}\n"
        "for line in sys.stdin:\n"
        "    if '\"prompt\"' in line:\n"
        "        print(json.dumps(msg), flush=True)\n"
        "        print(json.dumps({'type': 'agent_settled', 'aborted': False}), flush=True)\n"
        "        break\n",
        encoding="utf-8",
    )
    fake_pi.chmod(0o755)
    secret = "sk-ephemeral-gateway-secret-012345"
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _reset_pi_availability_cache()
    try:
        driver = PiDriver(binary=str(fake_pi))
        outcome = driver.run_turn(
            DriverTurnRequest(
                workspace_path=workspace,
                env_root=workspace,
                system_prompt="Return ok.",
                user_prompt="ping",
                actor="align:proposer",
                read_only=True,
                model="openai/custom-model-v1",
            )
        )
    finally:
        _reset_pi_availability_cache()

    assert outcome.completed is True
    captured = json.loads(capture_file.read_text(encoding="utf-8"))
    argv = captured["argv"]
    assert argv[argv.index("--provider") + 1] == "olympus-gateway"
    assert argv[argv.index("--model") + 1] == "custom-model-v1"
    assert secret not in captured["models_json"]
    doc = json.loads(captured["models_json"])
    provider_cfg = doc["providers"]["olympus-gateway"]
    assert provider_cfg["api"] == "openai-completions"
    assert provider_cfg["baseUrl"] == "https://gateway.example/v1"
    assert provider_cfg["apiKey"] == "$OPENAI_API_KEY"
    assert provider_cfg["models"] == [{"id": "custom-model-v1"}]
    assert not Path(captured["cfg_dir"]).exists()
    assert not (workspace / ".olympus").exists()


def test_execution_drivers_config_defaults_to_auto_and_accepts_legacy_snapshots() -> None:
    default = ExecutionDriversConfig()
    assert default.stage_driver == "auto"
    assert default.worker_driver == "auto"
    assert default.reviewer_driver == "auto"

    legacy = ExecutionDriversConfig.model_validate(
        {"worker_driver": "pi", "reviewer_driver": "native", "monitor_model": "monitor-x"}
    )
    assert legacy.stage_driver == "auto"
    assert legacy.worker_driver == "pi"
    assert legacy.reviewer_driver == "native"
    assert legacy.monitor_model == "monitor-x"


def test_execution_drivers_config_resolve_prefers_explicit_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLYMPUS_STAGE_DRIVER", "pi@env-stage")
    monkeypatch.delenv("OLYMPUS_WORKER_DRIVER", raising=False)
    monkeypatch.delenv("OLYMPUS_REVIEWER_DRIVER", raising=False)

    resolved = ExecutionDriversConfig.resolve()
    assert resolved.stage_driver == "pi@env-stage"
    assert resolved.worker_driver == "auto"

    explicit = ExecutionDriversConfig.resolve(stage_driver="native")
    assert explicit.stage_driver == "native"


def test_resolve_driver_spec_auto_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _StageProvider()
    mock = MockLLMProvider()
    fake = _write_fake_pi(tmp_path, rpc=True)
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(fake))
    _reset_pi_availability_cache()
    try:
        assert resolve_driver_spec("auto", provider=mock) == "native"
        assert resolve_driver_spec("auto", provider=object()) == "native"  # type: ignore[arg-type] -- custom test double stays native
        assert resolve_driver_spec("auto", provider=provider) == "pi@stage-model"
        assert resolve_driver_spec("auto@review-model", provider=provider) == "pi@review-model"
        assert resolve_driver_spec("pi", provider=provider) == "pi"
        assert resolve_driver_spec("pi@explicit", provider=provider) == "pi@explicit"
        assert resolve_driver_spec("native@local", provider=provider) == "native@local"

        monkeypatch.setenv("OLYMPUS_PI_BIN", str(tmp_path / "missing-pi"))
        _reset_pi_availability_cache()
        assert resolve_driver_spec("auto", provider=provider) == "native"
    finally:
        _reset_pi_availability_cache()


def test_argus_auto_selects_pi_with_provider_model_and_read_only_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    _init_repo(workspace)
    store, run_id = _confirmed_run(workspace)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    _approve_plan(store, run_id, _single_task_plan(spec, [{
        "id": "CHK-1",
        "kind": "independent_review",
        "description": "Cold-start review",
    }]))
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(_write_fake_pi(tmp_path, rpc=True)))
    _reset_pi_availability_cache()

    class RecordingPi(_ScriptedStageDriver):
        name = "pi"

    pi = RecordingPi([
        "Worker finished.",
        '{"task_id":"T1","attempt":1,"passed":true,"summary":"Verified","findings":[]}',
    ])
    provider = _StageProvider(model_name="argus-model")
    try:
        result = OrchestrateStage(
            store=store,
            provider=provider,
            workspace_root=workspace,
            execution_drivers=ExecutionDriversConfig(),
            driver_registry=DriverRegistry({"pi": pi}),
        ).run_execution(run_id=run_id)
    finally:
        _reset_pi_availability_cache()

    assert result["status"] == RunStatus.COMPLETED
    assert provider.calls == 0
    worker, reviewer = pi.requests
    assert worker.read_only is False
    assert reviewer.actor == "argus:T1"
    assert reviewer.model == "argus-model"
    assert reviewer.read_only is True


def test_align_stage_auto_runs_pi_read_only_and_leaves_workspace_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    state.mkdir()
    _init_repo(workspace)
    store = RunStore(state / "olympus.db")
    run_id = store.create_run(title="Auto align")
    fake = _write_fake_pi(state, rpc=True)
    payload_file = state / "align-payload.json"
    argv_file = state / "argv.json"
    payload_file.write_text(json.dumps(_stage_goal_payload()), encoding="utf-8")
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(fake))
    monkeypatch.setenv("FAKE_PI_PAYLOAD_FILE", str(payload_file))
    monkeypatch.setenv("FAKE_PI_ARGV_FILE", str(argv_file))
    monkeypatch.setenv("FAKE_PI_WRITE_FILE", str(state / "fake-write.txt"))
    _reset_pi_availability_cache()

    provider = _StageProvider()
    before = _workspace_files(workspace)
    stage = AlignStage(
        store=store,
        provider=provider,
        workspace_root=workspace,
        execution_drivers=ExecutionDriversConfig(stage_driver="auto"),
    )
    spec = stage.propose_spec(
        run_id=run_id,
        raw_goal="Expose a calculator module for developers",
        max_steps=6,
        strict_canonical=True,
    )
    _reset_pi_availability_cache()

    assert spec.status == SpecStatus.AWAITING_HUMAN
    assert spec.goal.startswith("Expose a calculator module")
    assert provider.calls == 0
    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert argv[argv.index("--tools") + 1] == "read,grep,find,ls"
    assert argv[argv.index("--exclude-tools") + 1] == "bash,write,edit,powershell"
    assert argv[argv.index("--model") + 1] == "stage-model"
    assert _workspace_files(workspace) == before
    assert _git_status_clean(workspace)
    assert not (workspace / ".olympus").exists()
    ledger = store.get_run(run_id)["budget_ledger"]
    assert (ledger.input_tokens, ledger.output_tokens) == (18, 8)
    saved = store.get_run(run_id)["spec"]
    assert saved is not None and saved.content_hash == spec.content_hash


def test_design_and_plan_stages_auto_run_pi_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    state.mkdir()
    _init_repo(workspace)
    store = RunStore(state / "olympus.db")
    run_id = store.create_run(title="Auto design and plan")
    spec = GoalSpec.model_validate(_stage_goal_payload())
    store.save_spec(run_id, spec)
    AlignStage(
        store=store, provider=MockLLMProvider(), workspace_root=workspace
    ).confirm_spec(run_id=run_id, approver="zeus")

    fake = _write_fake_pi(state, rpc=True)
    payload_file = state / "stage-payload.json"
    argv_file = state / "argv.json"
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(fake))
    monkeypatch.setenv("FAKE_PI_PAYLOAD_FILE", str(payload_file))
    monkeypatch.setenv("FAKE_PI_ARGV_FILE", str(argv_file))
    monkeypatch.setenv("FAKE_PI_WRITE_FILE", str(state / "fake-write.txt"))
    _reset_pi_availability_cache()

    provider = _StageProvider()
    drivers = ExecutionDriversConfig(stage_driver="auto")
    before = _workspace_files(workspace)

    payload_file.write_text(json.dumps(_stage_design_payload()), encoding="utf-8")
    design_stage = DesignStage(
        store=store,
        provider=provider,
        workspace_root=workspace,
        artifacts_root=state / "artifacts",
        execution_drivers=drivers,
    )
    design = design_stage.propose_design(run_id, max_steps=6)
    assert design.spec_hash == spec.content_hash
    assert design.architecture_summary.startswith("A single calculator module")
    design_stage.approve_design(run_id, approver="zeus")

    payload_file.write_text(json.dumps(_stage_plan_payload(spec)), encoding="utf-8")
    plan = OrchestrateStage(
        store=store,
        provider=provider,
        workspace_root=workspace,
        artifacts_root=state / "artifacts",
        execution_drivers=drivers,
    ).propose_plan(run_id=run_id, use_design=False, max_steps=6)
    _reset_pi_availability_cache()

    assert plan.spec_id == spec.spec_id
    assert plan.spec_hash == spec.content_hash
    assert [task.task_id for task in plan.tasks] == ["T1"]
    assert provider.calls == 0
    argv = json.loads(argv_file.read_text(encoding="utf-8"))["argv"]
    assert argv[argv.index("--tools") + 1] == "read,grep,find,ls"
    assert argv[argv.index("--exclude-tools") + 1] == "bash,write,edit,powershell"
    assert _workspace_files(workspace) == before
    assert _git_status_clean(workspace)
    assert not (workspace / ".olympus").exists()
    snap = store.get_run(run_id)
    assert snap["design"] is not None and snap["plan"] is not None


def test_stage_driver_retries_gate_violations_with_feedback_and_settles_usage(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    state.mkdir()
    _init_repo(workspace)
    store = RunStore(state / "olympus.db")
    run_id = store.create_run(title="Scripted stage retry")
    invalid = _stage_goal_payload()
    for key in ("deliverable_kind", "actors", "out_of_scope"):
        invalid.pop(key)
    driver = _ScriptedStageDriver([json.dumps(invalid), json.dumps(_stage_goal_payload())])
    provider = MockLLMProvider()
    stage = AlignStage(
        store=store,
        provider=provider,
        workspace_root=workspace,
        execution_drivers=ExecutionDriversConfig(stage_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": driver}),
    )

    spec = stage.propose_spec(
        run_id=run_id,
        raw_goal="Expose a calculator module for developers",
        strict_canonical=True,
    )

    assert len(driver.requests) == 2
    assert all(request.read_only for request in driver.requests)
    retry_prompt = driver.requests[1].user_prompt
    assert "[GOALSPEC VALIDATION FAILED]" in retry_prompt
    assert "deliverable_kind" in retry_prompt
    assert provider.recorded_requests == []
    assert spec.deliverable_kind is not None
    ledger = store.get_run(run_id)["budget_ledger"]
    assert (ledger.input_tokens, ledger.output_tokens) == (20, 10)
    assert ledger.llm_calls_used == 2


def test_stage_driver_exhaustion_records_validation_event(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    state.mkdir()
    _init_repo(workspace)
    store = RunStore(state / "olympus.db")
    run_id = store.create_run(title="Scripted stage exhaustion")
    invalid = _stage_goal_payload()
    for key in ("deliverable_kind", "actors", "out_of_scope"):
        invalid.pop(key)
    driver = _ScriptedStageDriver([json.dumps(invalid), json.dumps(invalid)])
    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(),
        workspace_root=workspace,
        execution_drivers=ExecutionDriversConfig(stage_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": driver}),
    )

    spec = stage.propose_spec(
        run_id=run_id,
        raw_goal="Expose a calculator module for developers",
        strict_canonical=True,
    )

    assert len(driver.requests) == 2
    assert spec.deliverable_kind is None
    events = [e for e in store.list_events(run_id) if e.event_type == "spec.validation_failed"]
    assert len(events) == 1
    payload = events[0].payload
    assert any("deliverable_kind" in violation for violation in payload["violations"])
    assert payload["spec_hash"] == spec.content_hash


def test_align_critics_run_through_stage_driver_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    state.mkdir()
    _init_repo(workspace)
    store = RunStore(state / "olympus.db")
    run_id = store.create_run(title="Scripted critique")
    critique = json.dumps(
        {
            "notes": [
                {
                    "id": "N1",
                    "perspective": "qa",
                    "severity": "warning",
                    "summary": "Zero-division coverage must be verified end to end",
                }
            ]
        }
    )
    driver = _ScriptedStageDriver([json.dumps(_stage_goal_payload())] + [critique] * 3)
    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(),
        workspace_root=workspace,
        execution_drivers=ExecutionDriversConfig(stage_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": driver}),
    )

    spec = stage.propose_spec(
        run_id=run_id,
        raw_goal="Expose a calculator module for developers",
        enable_critique=True,
    )

    assert len(driver.requests) == 4
    assert all(request.read_only for request in driver.requests)
    assert len(spec.minority_notes) == 3
    assert all(note.id == "N1" for note in spec.minority_notes)


def test_explicit_pi_stage_driver_fails_loudly_when_binary_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    state.mkdir()
    _init_repo(workspace)
    store = RunStore(state / "olympus.db")
    run_id = store.create_run(title="Explicit pi missing")
    monkeypatch.setenv("OLYMPUS_PI_BIN", str(tmp_path / "missing-pi"))
    _reset_pi_availability_cache()
    stage = AlignStage(
        store=store,
        provider=_StageProvider(),
        workspace_root=workspace,
        execution_drivers=ExecutionDriversConfig(stage_driver="pi"),
    )

    with pytest.raises(PiNotAvailableError) as excinfo:
        stage.propose_spec(run_id=run_id, raw_goal="Expose a calculator module")

    assert "not found" in str(excinfo.value)
    _reset_pi_availability_cache()


def test_stage_driver_honors_parse_retries_when_validation_retries_zero(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    state.mkdir()
    _init_repo(workspace)
    store = RunStore(state / "olympus.db")
    run_id = store.create_run(title="Parse retry")
    driver = _ScriptedStageDriver(["not-json-at-all", json.dumps(_stage_goal_payload())])
    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(),
        workspace_root=workspace,
        execution_drivers=ExecutionDriversConfig(stage_driver="scripted"),
        driver_registry=DriverRegistry({"scripted": driver}),
    )

    spec = stage.propose_spec(
        run_id=run_id,
        raw_goal="Expose a calculator module for developers",
        strict_canonical=False,
    )
    assert len(driver.requests) == 2
    assert driver.requests[0].disable_tools is False
    assert driver.requests[1].disable_tools is True
    assert spec.goal.startswith("Expose a calculator module")


def test_pi_driver_recovers_after_transient_error_message_retried_by_pi(tmp_path: Path) -> None:
    fake = tmp_path / "fake-pi-retry"
    fake.write_text(
        """#!/usr/bin/env python3
import json, sys
if "--help" in sys.argv:
    print("--mode json rpc")
    sys.exit(0)
for raw in sys.stdin:
    cmd = json.loads(raw)
    if cmd.get("type") == "prompt":
        print(json.dumps({"type": "response", "command": "prompt", "success": True}), flush=True)
        print(json.dumps({
            "type": "message_end",
            "message": {"role": "assistant", "stopReason": "error", "content": [], "usage": {}},
        }), flush=True)
        print(json.dumps({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "content": [{"type": "text", "text": '{"ok": true}'}],
                "usage": {"input": 10, "output": 5},
            },
        }), flush=True)
        print(json.dumps({"type": "agent_settled", "aborted": False}), flush=True)
        break
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    outcome = PiDriver(binary=str(fake)).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="align:proposer",
            system_prompt="",
            user_prompt="test",
            read_only=True,
        )
    )
    assert outcome.completed is True
    assert outcome.stop_reason == "final_response"
    assert outcome.final_content == '{"ok": true}'


def test_auto_driver_rejects_unconfigured_mock_and_invalidates_path_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    from olympus.harness.driver import resolve_driver_spec
    from olympus.harness.pi import pi_available

    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    good_dir.mkdir()
    bad_dir.mkdir()
    good_pi = good_dir / "pi"
    good_pi.write_text("#!/bin/sh\necho '--mode json rpc'\n", encoding="utf-8")
    good_pi.chmod(0o755)
    bad_pi = bad_dir / "pi"
    bad_pi.write_text("#!/bin/sh\necho 'usage: pi'\n", encoding="utf-8")
    bad_pi.chmod(0o755)

    _reset_pi_availability_cache()
    monkeypatch.delenv("OLYMPUS_PI_BIN", raising=False)
    monkeypatch.setenv("PATH", f"{good_dir}:{os.environ.get('PATH', '')}")
    assert pi_available() is True
    assert resolve_driver_spec("auto", provider=Mock()) == "native"

    monkeypatch.setenv("PATH", f"{bad_dir}:{os.environ.get('PATH', '')}")
    assert pi_available() is False
    _reset_pi_availability_cache()


def test_pi_driver_provider_binding_secret_redaction_and_error_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.providers.factory import build_provider

    captured = tmp_path / "models.json"
    trace = tmp_path / "pi-trace.jsonl"
    fake = tmp_path / "fake-pi-gw"
    fake.write_text(
        f"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if "--help" in sys.argv:
    print("--mode json rpc")
    sys.exit(0)
cfg = Path(os.environ["PI_CODING_AGENT_DIR"]) / "models.json"
Path({str(captured)!r}).write_text(cfg.read_text(), encoding="utf-8")
sys.stdin.readline()
key = os.environ["OLYMPUS_PI_GATEWAY_KEY"]
print("provider read timed out: " + key, file=sys.stderr, flush=True)
print(json.dumps({{
    "type": "message_end",
    "message": {{
        "role": "assistant",
        "stopReason": "error",
        "errorMessage": "provider read timed out",
    }},
}}), flush=True)
print(json.dumps({{"type": "agent_settled", "aborted": False}}), flush=True)
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = {
        "OPENAI_BASE_URL": "https://ambient-openai.invalid/v1",
        "OPENAI_API_KEY": "sk-ambient-openai-sentinel-999",
        "ANTHROPIC_BASE_URL": "https://ambient-anthropic.invalid",
        "ANTHROPIC_API_KEY": "sk-selected-anthropic-sentinel-888",
        "PI_CODING_AGENT_DIR": "",
    }
    provider = build_provider(
        provider="anthropic",
        model="custom-model",
        base_url="https://selected-gateway.invalid",
        environ=env,
        user_providers_path=None,
    )
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    outcome = PiDriver(binary=str(fake), provider=provider).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="plan",
            system_prompt="Return JSON",
            user_prompt="Plan",
            read_only=True,
            model=provider.model_name,
            trace_path=trace,
        )
    )
    cfg = json.loads(captured.read_text(encoding="utf-8"))["providers"]["olympus-gateway"]
    assert cfg["baseUrl"] == "https://selected-gateway.invalid"
    assert cfg["api"] == "anthropic-messages"
    assert cfg["apiKey"] == "$OLYMPUS_PI_GATEWAY_KEY"
    assert "sk-selected-anthropic-sentinel-888" not in captured.read_text(encoding="utf-8")
    assert "sk-selected-anthropic-sentinel-888" not in trace.read_text(encoding="utf-8")
    assert outcome.completed is False
    assert "pi_error" in outcome.stop_reason
    assert "timed out" in outcome.stop_reason


def test_pi_driver_cleans_up_gateway_dir_on_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    from olympus.harness.pi import pi_available

    fake = tmp_path / "vanishing-pi"
    fake.write_text("#!/bin/sh\necho '--mode json rpc'\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-vanish-sentinel-777")
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    driver = PiDriver(binary=str(fake))
    # Prime the help probe, then remove the binary before spawn.
    assert pi_available(binary=str(fake)) is True
    fake.unlink()
    before = set(Path(tempfile.gettempdir()).glob("olympus-pi-cfg-*"))
    with pytest.raises(PiNotAvailableError):
        driver.run_turn(
            DriverTurnRequest(
                workspace_path=tmp_path,
                env_root=tmp_path,
                actor="plan",
                system_prompt="",
                user_prompt="hi",
                read_only=True,
                model="openai/gpt-4o",
            )
        )
    after = set(Path(tempfile.gettempdir()).glob("olympus-pi-cfg-*"))
    assert after - before == set()
    _reset_pi_availability_cache()


_GATEWAY_PROBE_SOURCE = '''
"""Fake Pi CLI capturing gateway config, argv, and file modes for Olympus tests."""
import json
import os
import stat
import sys
from pathlib import Path

CAPTURE_FILE = os.environ["FAKE_PI_CAPTURE_FILE"]


def _main():
    argv = sys.argv[1:]
    if "--help" in argv:
        print("  --mode <mode>  Output mode: text, json, or rpc")
        return 0
    cfg_dir = os.environ.get("PI_CODING_AGENT_DIR", "")
    models_path = Path(cfg_dir) / "models.json" if cfg_dir else None
    payload = {
        "argv": argv,
        "cfg_dir": cfg_dir,
        "models_json": models_path.read_text(encoding="utf-8") if models_path else None,
        "cfg_dir_mode": stat.S_IMODE(Path(cfg_dir).stat().st_mode) if cfg_dir else None,
        "models_mode": (
            stat.S_IMODE(models_path.stat().st_mode) if models_path is not None else None
        ),
        "pi_offline": os.environ.get("PI_OFFLINE", ""),
        "pi_skip_version_check": os.environ.get("PI_SKIP_VERSION_CHECK", ""),
    }
    Path(CAPTURE_FILE).write_text(json.dumps(payload), encoding="utf-8")
    for line in sys.stdin:
        if '"prompt"' in line:
            reasoning = int(os.environ.get("FAKE_PI_REASONING", "0"))
            print(json.dumps({
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input": 5, "output": 2, "reasoning": reasoning},
                },
            }), flush=True)
            print(json.dumps({"type": "agent_settled", "aborted": False}), flush=True)
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
'''


def _write_gateway_probe(root: Path) -> Path:
    path = root / "gateway_probe.py"
    path.write_text(f"#!{sys.executable}\n" + _GATEWAY_PROBE_SOURCE, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_pi_driver_native_provider_passthrough_with_thinking_and_no_models_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    capture_file = tmp_path / "pi-capture.json"
    fake = _write_gateway_probe(tmp_path)
    monkeypatch.setenv("FAKE_PI_CAPTURE_FILE", str(capture_file))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    _reset_pi_availability_cache()
    try:
        driver = PiDriver(binary=str(fake), pi_provider="gateway-x", thinking_level="HIGH")
        outcome = driver.run_turn(
            DriverTurnRequest(
                workspace_path=workspace,
                env_root=workspace,
                system_prompt="Return ok.",
                user_prompt="ping",
                actor="align:proposer",
                read_only=True,
                model="gateway-x/gateway-model",
            )
        )
    finally:
        _reset_pi_availability_cache()

    assert outcome.completed is True
    captured = json.loads(capture_file.read_text(encoding="utf-8"))
    argv = captured["argv"]
    assert argv[argv.index("--provider") + 1] == "gateway-x"
    assert argv[argv.index("--model") + 1] == "gateway-model"
    assert argv[argv.index("--thinking") + 1] == "high"
    # Pure passthrough: Pi's own provider catalog supplies the wiring, no models.json.
    assert captured["cfg_dir"] == ""
    assert captured["models_json"] is None
    assert not (workspace / ".olympus").exists()


def test_pi_driver_provider_override_models_json_with_base_url_and_context_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    capture_file = tmp_path / "pi-capture.json"
    fake = _write_gateway_probe(tmp_path)
    secret = "sk-passthrough-gateway-secret-424242"
    monkeypatch.setenv("FAKE_PI_CAPTURE_FILE", str(capture_file))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _reset_pi_availability_cache()
    try:
        driver = PiDriver(binary=str(fake), pi_provider="openai", context_window=262144)
        outcome = driver.run_turn(
            DriverTurnRequest(
                workspace_path=workspace,
                env_root=workspace,
                system_prompt="Return ok.",
                user_prompt="ping",
                actor="align:proposer",
                read_only=True,
                model="openai/gpt-4o",
            )
        )
    finally:
        _reset_pi_availability_cache()

    assert outcome.completed is True
    captured = json.loads(capture_file.read_text(encoding="utf-8"))
    argv = captured["argv"]
    assert argv[argv.index("--provider") + 1] == "openai"
    assert argv[argv.index("--model") + 1] == "gpt-4o"
    raw = captured["models_json"]
    assert raw is not None and secret not in raw
    doc = json.loads(raw)
    override = doc["providers"]["openai"]
    # Provider-override shape: no "models" array, modelOverrides carries the window.
    assert "models" not in override
    assert override == {
        "baseUrl": "https://gateway.example/v1",
        "apiKey": "$OPENAI_API_KEY",
        "modelOverrides": {"gpt-4o": {"contextWindow": 262144}},
    }
    assert captured["cfg_dir_mode"] == 0o700
    assert captured["models_mode"] == 0o600
    assert captured["pi_offline"] == "1"
    assert captured["pi_skip_version_check"] == "1"
    # The ephemeral config dir is removed once the turn reaps.
    assert not Path(captured["cfg_dir"]).exists()


def test_pi_driver_respects_caller_pi_coding_agent_dir_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    caller_dir = tmp_path / "caller-pi-cfg"
    caller_dir.mkdir()
    sentinel = '{"providers": {"caller-owned": {"baseUrl": "https://caller.invalid"}}}'
    (caller_dir / "models.json").write_text(sentinel, encoding="utf-8")
    capture_file = tmp_path / "pi-capture.json"
    fake = _write_gateway_probe(tmp_path)
    monkeypatch.setenv("FAKE_PI_CAPTURE_FILE", str(capture_file))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(caller_dir))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _reset_pi_availability_cache()
    try:
        driver = PiDriver(binary=str(fake), pi_provider="gateway-x", context_window=200000)
        outcome = driver.run_turn(
            DriverTurnRequest(
                workspace_path=workspace,
                env_root=workspace,
                system_prompt="Return ok.",
                user_prompt="ping",
                actor="align:proposer",
                read_only=True,
                model="gateway-x/gateway-model",
            )
        )
    finally:
        _reset_pi_availability_cache()

    assert outcome.completed is True
    captured = json.loads(capture_file.read_text(encoding="utf-8"))
    argv = captured["argv"]
    assert argv[argv.index("--provider") + 1] == "gateway-x"
    # Caller-owned config dir wins: the model passes through unstripped, nothing overwritten.
    assert argv[argv.index("--model") + 1] == "gateway-x/gateway-model"
    assert captured["cfg_dir"] == str(caller_dir)
    assert captured["models_json"] == sentinel
    assert (caller_dir / "models.json").read_text(encoding="utf-8") == sentinel


def test_pi_driver_legacy_gateway_models_json_carries_context_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    capture_file = tmp_path / "pi-capture.json"
    fake = _write_gateway_probe(tmp_path)
    secret = "sk-legacy-gateway-secret-313131"
    monkeypatch.setenv("FAKE_PI_CAPTURE_FILE", str(capture_file))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _reset_pi_availability_cache()
    try:
        driver = PiDriver(binary=str(fake), context_window=131072)
        outcome = driver.run_turn(
            DriverTurnRequest(
                workspace_path=workspace,
                env_root=workspace,
                system_prompt="Return ok.",
                user_prompt="ping",
                actor="align:proposer",
                read_only=True,
                model="openai/custom-model-v1",
            )
        )
    finally:
        _reset_pi_availability_cache()

    assert outcome.completed is True
    captured = json.loads(capture_file.read_text(encoding="utf-8"))
    argv = captured["argv"]
    assert argv[argv.index("--provider") + 1] == "olympus-gateway"
    assert argv[argv.index("--model") + 1] == "custom-model-v1"
    raw = captured["models_json"]
    assert raw is not None and secret not in raw
    doc = json.loads(raw)
    gateway = doc["providers"]["olympus-gateway"]
    assert gateway["api"] == "openai-completions"
    assert gateway["baseUrl"] == "https://gateway.example/v1"
    assert gateway["apiKey"] == "$OPENAI_API_KEY"
    assert gateway["models"] == [{"id": "custom-model-v1", "contextWindow": 131072}]
    assert not Path(captured["cfg_dir"]).exists()


def test_execution_drivers_config_validates_thinking_and_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLYMPUS_PI_PROVIDER", raising=False)
    monkeypatch.delenv("OLYMPUS_THINKING_LEVEL", raising=False)
    monkeypatch.delenv("OLYMPUS_CONTEXT_WINDOW", raising=False)

    resolved = ExecutionDriversConfig.resolve(
        pi_provider="gateway-nine", thinking_level="XHIGH", context_window=262144
    )
    assert resolved.pi_provider == "gateway-nine"
    assert resolved.thinking_level == "xhigh"
    assert resolved.context_window == 262144

    monkeypatch.setenv("OLYMPUS_PI_PROVIDER", "env-gateway")
    monkeypatch.setenv("OLYMPUS_THINKING_LEVEL", "Medium")
    monkeypatch.setenv("OLYMPUS_CONTEXT_WINDOW", "200000")
    from_env = ExecutionDriversConfig.resolve()
    assert from_env.pi_provider == "env-gateway"
    assert from_env.thinking_level == "medium"
    assert from_env.context_window == 200000
    # Explicit args still win over the environment.
    explicit = ExecutionDriversConfig.resolve(
        pi_provider="cli-gateway", thinking_level="low", context_window=120000
    )
    assert (explicit.pi_provider, explicit.thinking_level, explicit.context_window) == (
        "cli-gateway",
        "low",
        120000,
    )

    with pytest.raises(ValueError):
        ExecutionDriversConfig.resolve(thinking_level="extreme")
    from olympus.harness.pi import _normalize_thinking_level

    with pytest.raises(ValueError):
        _normalize_thinking_level("ultra")
    with pytest.raises(ValueError):
        monkeypatch.setenv("OLYMPUS_CONTEXT_WINDOW", "not-a-number")
        ExecutionDriversConfig.resolve()
    monkeypatch.setenv("OLYMPUS_CONTEXT_WINDOW", "512")
    with pytest.raises(ValueError):
        ExecutionDriversConfig.resolve()


def test_pi_driver_propagates_reasoning_and_cache_totals_to_receipt_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.harness.receipt import build_turn_receipt

    monkeypatch.setenv("FAKE_PI_CACHE_READ", "40")
    monkeypatch.setenv("FAKE_PI_CACHE_WRITE", "5")
    monkeypatch.setenv("FAKE_PI_REASONING", "7")
    store = RunStore(tmp_path / "run.db")
    run_id = store.create_run(
        title="reasoning-accounting",
        mode=RunMode.SIMULATION,
        budget_policy=BudgetPolicy(max_total_tokens=10_000),
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

    assert outcome.completed is True
    assert outcome.steps_taken == 2
    # Two assistant messages: (11,5) toolUse + (7,3) final, cache/reasoning each.
    assert (outcome.total_input_tokens, outcome.total_output_tokens) == (18, 8)
    assert (outcome.total_cache_read_tokens, outcome.total_cache_write_tokens) == (80, 10)
    assert outcome.total_reasoning_tokens == 14

    llm_events = [payload for kind, payload in events if kind == "llm"]
    assert len(llm_events) == 2
    assert all(payload["reasoning_tokens"] == 7 for payload in llm_events)
    assert all(payload["cache_read_tokens"] == 40 for payload in llm_events)

    llm = outcome.receipt["llm"]
    assert llm == {
        "calls": 2,
        "input_tokens": 18,
        "output_tokens": 8,
        "cache_read_tokens": 80,
        "cache_write_tokens": 10,
        "reasoning_tokens": 14,
        "total_tokens": 116,
    }
    # Reasoning is a non-additive subset of output tokens: never double-counted.
    assert llm["total_tokens"] == 18 + 8 + 80 + 10
    assert outcome.receipt == build_turn_receipt(
        turn_id=outcome.receipt["turn_id"],
        actor="worker",
        events=[
            {
                "kind": "llm",
                "input_tokens": 11,
                "output_tokens": 5,
                "cache_read_tokens": 40,
                "cache_write_tokens": 5,
                "reasoning_tokens": 7,
            },
            {
                "kind": "llm",
                "input_tokens": 7,
                "output_tokens": 3,
                "cache_read_tokens": 40,
                "cache_write_tokens": 5,
                "reasoning_tokens": 7,
            },
            {
                "kind": "tool_end",
                "tool": "write",
                "is_error": False,
                "duration_ms": 12,
            },
        ],
    )

    ledger = store.get_run(run_id)["budget_ledger"]
    assert (ledger.input_tokens, ledger.output_tokens) == (18, 8)
    assert (ledger.cache_read_tokens, ledger.cache_write_tokens) == (80, 10)
    assert ledger.reasoning_tokens == 14
    assert ledger.total_tokens == 116
    assert ledger.cache_hit_rate == pytest.approx(round(80 / (18 + 80), 4))

    llm_trace = [entry for entry in _read_jsonl(trace_path) if entry.get("event") == "llm"]
    assert all(entry["reasoning_tokens"] == 7 for entry in llm_trace)

    settled = [
        event.payload
        for event in store.list_events(run_id)
        if event.event_type == "budget.settled"
    ]
    assert len(settled) == 2
    assert all(payload["reasoning_tokens"] == 7 for payload in settled)


def test_design_and_orchestrate_default_artifacts_stay_outside_repo(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    state = tmp_path / "state"
    state.mkdir()
    _init_repo(workspace)
    store = RunStore(state / "olympus.db")
    design = DesignStage(store=store, provider=MockLLMProvider(), workspace_root=workspace)
    orch = OrchestrateStage(store=store, provider=MockLLMProvider(), workspace_root=workspace)
    assert design.artifacts_root == state / "runs"
    assert orch.artifacts_root == state / "runs"
    assert not (workspace / ".olympus").exists()


def test_pi_driver_redacts_milestone_records_and_preserves_draft_on_max_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import BaseModel

    from olympus.harness.loop import AgentLoop
    from olympus.harness.structured import run_structured_turn

    class _Payload(BaseModel):
        result: str

    secret = "sk-milestone-trace-sentinel-98765"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    trace = tmp_path / "trace.jsonl"
    capture = tmp_path / "cap-prompt.txt"
    fake = tmp_path / "fake-pi-redact-cap"
    fake.write_text(
        f"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if "--help" in sys.argv:
    print("--mode json")
    sys.exit(0)
def emit(reason, text="", **extra):
    msg = {{"role": "assistant", "stopReason": reason,
            "content": [{{"type": "text", "text": text}}], **extra}}
    print(json.dumps({{"type": "message_end", "message": msg}}), flush=True)
if sys.argv[-1] == "redact":
    emit("error", errorMessage="gateway error: " + os.environ["OPENAI_API_KEY"])
elif "--no-tools" in sys.argv:
    Path({str(capture)!r}).write_text(sys.argv[-1], encoding="utf-8")
    verdict = "grounded" if "GROUNDED_DRAFT" in sys.argv[-1] else "bad"
    emit("stop", json.dumps({{"result": verdict}}))
else:
    emit("toolUse", "GROUNDED_DRAFT")
    emit("toolUse", "")
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    driver = PiDriver(binary=str(fake))

    err_outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="plan",
            system_prompt="sys",
            user_prompt="redact",
            read_only=True,
            max_steps=1,
            trace_path=trace,
        )
    )
    assert err_outcome.completed is False
    assert secret not in err_outcome.stop_reason
    assert secret not in trace.read_text(encoding="utf-8")

    result = run_structured_turn(
        loop=AgentLoop(provider=MockLLMProvider()),
        driver=driver,
        driver_model=None,
        store=None,
        run_id=None,
        workspace_root=tmp_path,
        actor="plan",
        system_prompt="Return JSON",
        user_prompt="Extract facts",
        model_cls=_Payload,
        max_steps=1,
    )
    assert "GROUNDED_DRAFT" in capture.read_text(encoding="utf-8")
    assert result.result == "grounded"


def test_pi_driver_native_provider_ignores_builtin_default_endpoint_and_preserves_openrouter_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.providers.factory import build_provider

    workspace = tmp_path / "repo"
    workspace.mkdir()
    capture_file = tmp_path / "capture.json"
    fake = tmp_path / "fake-pi-native-check"
    fake.write_text(
        f"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if "--help" in sys.argv:
    print("--mode json")
    sys.exit(0)
cfg_dir = os.environ.get("PI_CODING_AGENT_DIR")
auth_link = None
models_json = None
if cfg_dir:
    p = Path(cfg_dir)
    if (p / "models.json").is_file():
        models_json = (p / "models.json").read_text(encoding="utf-8")
    if (p / "auth.json").is_symlink():
        auth_link = str(os.readlink(p / "auth.json"))
Path({str(capture_file)!r}).write_text(
    json.dumps({{
        "argv": sys.argv[1:],
        "cfg_dir": cfg_dir,
        "models_json": models_json,
        "auth_link": auth_link,
    }}),
    encoding="utf-8",
)
msg = {{"role": "assistant", "stopReason": "stop", "content": [{{"type": "text", "text": "ok"}}],
        "usage": {{"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0}}}}
print(json.dumps({{"type": "message_end", "message": msg}}), flush=True)
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    fake_home = tmp_path / "home"
    ambient_pi = fake_home / ".pi" / "agent"
    ambient_pi.mkdir(parents=True)
    ambient_auth = ambient_pi / "auth.json"
    ambient_auth.write_text('{"openrouter": {"apiKey": "saved-key"}}\n', encoding="utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLYMPUS_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-openai-12345678")

    # Built-in default OpenAI provider (no --base-url) must NOT hijack a native pi_provider.
    builtin_provider = build_provider(provider="openai", model="openai/gpt-4.1")
    _reset_pi_availability_cache()
    try:
        driver = PiDriver(
            binary=str(fake),
            provider=builtin_provider,
            pi_provider="openrouter",
            thinking_level="high",
            context_window=262144,
        )
        outcome = driver.run_turn(
            DriverTurnRequest(
                workspace_path=workspace,
                env_root=workspace,
                system_prompt="",
                user_prompt="hi",
                actor="worker",
                read_only=True,
                model="openai/gpt-4.1",
            )
        )
    finally:
        _reset_pi_availability_cache()

    assert outcome.completed is True
    captured = json.loads(capture_file.read_text(encoding="utf-8"))
    argv = captured["argv"]
    assert argv[argv.index("--provider") + 1] == "openrouter"
    # Finding #3: OpenRouter catalog ID 'openai/gpt-4.1' must NOT lose 'openai/'.
    assert argv[argv.index("--model") + 1] == "openai/gpt-4.1"
    # Finding #1: Built-in OpenAI endpoint must NOT be written into openrouter override.
    doc = json.loads(captured["models_json"])
    override = doc["providers"]["openrouter"]
    assert "baseUrl" not in override
    # Finding #2: Ambient ~/.pi/agent/auth.json is read in memory via $OLYMPUS_PI_GATEWAY_KEY
    # without symlinking or serializing the credential to disk, so Pi cannot mutate ~/.pi/agent.
    assert override["apiKey"] == "$OLYMPUS_PI_GATEWAY_KEY"
    assert "saved-key" not in captured["models_json"]
    assert captured["auth_link"] is None
    assert override["modelOverrides"] == {"openai/gpt-4.1": {"contextWindow": 262144}}
    assert ambient_auth.is_file()
    assert not Path(captured["cfg_dir"]).exists()

    # Finding #1 regression: providers.json overriding built-in 'openai' with a custom base_url
    # MUST still be treated as a custom endpoint and passed into Pi's provider override.
    providers_cfg = tmp_path / "providers.json"
    providers_cfg.write_text(
        json.dumps(
            {
                "providers": {
                    "openai": {
                        "dialect": "openai-chat",
                        "default_model": "custom-model-v1",
                        "base_url": "https://custom-proxy.example/v1",
                        "api_key_env": "OPENAI_API_KEY",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    custom_from_file = build_provider(
        provider="openai",
        model="custom-model-v1",
        providers_file=providers_cfg,
    )
    binding = custom_from_file.external_gateway_binding("custom-model-v1")
    assert binding is not None
    assert binding.is_custom_endpoint is True
    assert binding.base_url == "https://custom-proxy.example/v1"

    # Finding #2 regression: provider-level env map in ~/.pi/agent/auth.json is forwarded
    # into subprocess env (and redacted) when credential references $VAR.
    ambient_auth.write_text(
        json.dumps(
            {
                "openrouter": {
                    "type": "api_key",
                    "key": "$AMBIENT_PROV_KEY",
                    "env": {"AMBIENT_PROV_KEY": "sk-ambient-env-secret"},
                }
            }
        ),
        encoding="utf-8",
    )
    from olympus.harness.pi import _prepare_gateway_config

    probe_env: dict[str, str] = {"HOME": str(fake_home)}
    prov_name, mod_id, cfg_dir, extra_secrets = _prepare_gateway_config(
        probe_env,
        "openai/gpt-4.1",
        provider=builtin_provider,
        pi_provider="openrouter",
        context_window=262144,
    )
    try:
        assert prov_name == "openrouter"
        assert mod_id == "openai/gpt-4.1"
        assert probe_env["AMBIENT_PROV_KEY"] == "sk-ambient-env-secret"
        assert "sk-ambient-env-secret" in extra_secrets
        assert cfg_dir is not None
        doc2 = json.loads((cfg_dir / "models.json").read_text(encoding="utf-8"))
        assert doc2["providers"]["openrouter"]["apiKey"] == "$AMBIENT_PROV_KEY"
        assert "sk-ambient-env-secret" not in (cfg_dir / "models.json").read_text(
            encoding="utf-8"
        )
    finally:
        if cfg_dir is not None:
            shutil.rmtree(cfg_dir, ignore_errors=True)


def test_pi_driver_redacts_short_and_custom_env_binding_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.providers.auth import ExternalGatewayBinding

    class _CustomBindingProvider:
        model_name = "demo-model"

        def external_gateway_binding(self, model: str | None = None) -> ExternalGatewayBinding:
            del model
            return ExternalGatewayBinding(
                api="openai-completions",
                base_url="https://custom.example/v1",
                model_id="demo-model",
                api_key_env="MY_CUSTOM_CRED_VAR",
                is_custom_endpoint=True,
            )

    short_secret = "k9x2z"
    monkeypatch.setenv("MY_CUSTOM_CRED_VAR", short_secret)
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    trace = tmp_path / "trace.jsonl"
    fake = tmp_path / "fake-pi-short-secret"
    fake.write_text(
        """#!/usr/bin/env python3
import json, os, sys
if "--help" in sys.argv:
    print("--mode json")
    sys.exit(0)
sec = os.environ["MY_CUSTOM_CRED_VAR"]
print("stderr leak " + sec, file=sys.stderr, flush=True)
msg = {
    "role": "assistant",
    "stopReason": "error",
    "errorMessage": "auth failed: " + sec,
    "content": [],
}
print(json.dumps({"type": "message_end", "message": msg}), flush=True)
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    _reset_pi_availability_cache()
    try:
        outcome = PiDriver(
            binary=str(fake),
            provider=_CustomBindingProvider(),
            pi_provider="custom-prov",
        ).run_turn(
            DriverTurnRequest(
                workspace_path=tmp_path,
                env_root=tmp_path,
                actor="worker",
                system_prompt="",
                user_prompt="hi",
                read_only=True,
                model="demo-model",
                trace_path=trace,
            )
        )
    finally:
        _reset_pi_availability_cache()

    assert outcome.completed is False
    assert short_secret not in outcome.stop_reason
    assert short_secret not in trace.read_text(encoding="utf-8")

    # Finding #3 regression: even a literal secret value equal to 'keyless' must be redacted.
    monkeypatch.setenv("MY_CUSTOM_CRED_VAR", "keyless")
    trace2 = tmp_path / "trace-keyless.jsonl"
    _reset_pi_availability_cache()
    try:
        outcome2 = PiDriver(
            binary=str(fake),
            provider=_CustomBindingProvider(),
            pi_provider="custom-prov",
        ).run_turn(
            DriverTurnRequest(
                workspace_path=tmp_path,
                env_root=tmp_path,
                actor="worker",
                system_prompt="",
                user_prompt="hi",
                read_only=True,
                model="demo-model",
                trace_path=trace2,
            )
        )
    finally:
        _reset_pi_availability_cache()
    assert outcome2.completed is False
    assert "keyless" not in outcome2.stop_reason
    assert "keyless" not in trace2.read_text(encoding="utf-8")


def test_pi_env_injects_node_keepalive_preload_preserving_existing_node_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.harness.pi import _NET_KEEPALIVE_PATH, _build_pi_env

    assert _NET_KEEPALIVE_PATH.is_file()
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=4096")
    env = _build_pi_env(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="",
            user_prompt="hi",
            read_only=False,
        )
    )
    node_opts = env.get("NODE_OPTIONS", "")
    assert "--max-old-space-size=4096" in node_opts
    assert f"--require={_NET_KEEPALIVE_PATH.resolve()}" in node_opts

    # Idempotent when already present.
    monkeypatch.setenv("NODE_OPTIONS", node_opts)
    env2 = _build_pi_env(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="",
            user_prompt="hi",
            read_only=False,
        )
    )
    assert env2["NODE_OPTIONS"].count(str(_NET_KEEPALIVE_PATH.resolve())) == 1
