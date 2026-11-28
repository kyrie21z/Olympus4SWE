"""Offline OpenCodeDriver protocol tests and contract verification.

Every test drives a deterministic local fake ``opencode`` Python script:
zero network and zero real LLM calls.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from threading import Barrier, Event, Thread
from typing import Any

import pytest

from olympus.core.contracts import BudgetPolicy, RunMode
from olympus.core.store import RunStore
from olympus.harness.driver import (
    ActiveTurnHandle,
    DriverRegistry,
    DriverTurnRequest,
    ExecutionDriversConfig,
    default_driver_factories,
    resolve_driver_spec,
)
from olympus.harness.opencode import (
    OpenCodeDriver,
    OpenCodeNotAvailableError,
    _build_agent_permissions,
    _reset_opencode_availability_cache,
    _safe_argv,
    opencode_available,
)
from olympus.providers.auth import ExternalGatewayBinding
from olympus.providers.base import MockLLMProvider

_FAKE_OPENCODE_SOURCE = '''
"""Deterministic fake OpenCode CLI for offline Olympus unit tests."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time


def _emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\\n")
    sys.stdout.flush()


def _write_json(env_key: str, payload: object) -> None:
    path = os.environ.get(env_key)
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "run" and "--help" in argv:
        if os.environ.get("FAKE_OC_NO_JSON_HELP") == "1":
            sys.stdout.write("USAGE\\n  opencode run [flags]\\nFLAGS\\n  --model string\\n")
        else:
            sys.stdout.write(
                "USAGE\\n  opencode run [flags]\\nFLAGS\\n"
                "  --format choice  Output format (choices: default, json)\\n"
            )
        return 0

    _write_json(
        "FAKE_OC_CAPTURE_PATH",
        {
            "argv": argv,
            "cwd": os.getcwd(),
            "pwd": os.environ.get("PWD"),
            "config": json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", "{}")),
            "xdg_data": os.environ.get("XDG_DATA_HOME"),
            "xdg_config": os.environ.get("XDG_CONFIG_HOME"),
            "xdg_state": os.environ.get("XDG_STATE_HOME"),
            "opencode_db": os.environ.get("OPENCODE_DB"),
            "opencode_config": os.environ.get("OPENCODE_CONFIG"),
            "opencode_config_dir": os.environ.get("OPENCODE_CONFIG_DIR"),
            "opencode_permission": os.environ.get("OPENCODE_PERMISSION"),
            "disable_claude": os.environ.get("OPENCODE_DISABLE_CLAUDE_CODE"),
            "disable_project_config": os.environ.get("OPENCODE_DISABLE_PROJECT_CONFIG"),
            "config_project_disable": os.environ.get("OPENCODE_CONFIG_PROJECT_DISABLE"),
            "gateway_key_present": bool(os.environ.get("OLYMPUS_OPENCODE_GATEWAY_KEY")),
        },
    )

    scenario = os.environ.get("FAKE_OC_SCENARIO", "happy")

    if scenario == "happy":
        _emit({"type": "step_start", "sessionID": "ses_1", "part": {"type": "step-start"}})
        _emit(
            {
                "type": "text",
                "sessionID": "ses_1",
                "part": {"type": "text", "text": "Checking file..."},
            }
        )
        _emit(
            {
                "type": "tool_use",
                "sessionID": "ses_1",
                "part": {
                    "id": "call_1",
                    "type": "tool",
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"path": "README.md"},
                        "output": "hello world",
                        "time": {"start": 1000, "end": 1045},
                    },
                },
            }
        )
        _emit(
            {
                "type": "step_finish",
                "sessionID": "ses_1",
                "part": {
                    "type": "step-finish",
                    "reason": "tool-calls",
                    "tokens": {
                        "input": 100,
                        "output": 20,
                        "reasoning": 5,
                        "cache": {"read": 30, "write": 10},
                    },
                },
            }
        )
        _emit({"type": "step_start", "sessionID": "ses_1", "part": {"type": "step-start"}})
        _emit(
            {
                "type": "text",
                "sessionID": "ses_1",
                "part": {"type": "text", "text": "Final answer OK"},
            }
        )
        _emit(
            {
                "type": "step_finish",
                "sessionID": "ses_1",
                "part": {
                    "type": "step-finish",
                    "reason": "stop",
                    "tokens": {
                        "input": 50,
                        "output": 15,
                        "reasoning": 2,
                        "cache": {"read": 40, "write": 0},
                    },
                },
            }
        )
        return 0

    if scenario == "permission_denied":
        _emit({"type": "step_start", "sessionID": "ses_2", "part": {"type": "step-start"}})
        _emit(
            {
                "type": "tool_use",
                "sessionID": "ses_2",
                "part": {
                    "id": "call_deny",
                    "type": "tool",
                    "tool": "edit",
                    "state": {
                        "status": "error",
                        "input": {"path": "blocked.txt"},
                        "error": "Permission denied: edit",
                        "time": {"start": 2000, "end": 2012},
                    },
                },
            }
        )
        _emit(
            {
                "type": "text",
                "sessionID": "ses_2",
                "part": {"type": "text", "text": "Blocked by policy."},
            }
        )
        _emit(
            {
                "type": "step_finish",
                "sessionID": "ses_2",
                "part": {
                    "type": "step-finish",
                    "reason": "stop",
                    "tokens": {
                        "input": 40,
                        "output": 10,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
            }
        )
        return 0

    if scenario == "multi_step_loop":
        for idx in range(5):
            _emit({"type": "step_start", "sessionID": "ses_3", "part": {"type": "step-start"}})
            _emit(
                {
                    "type": "step_finish",
                    "sessionID": "ses_3",
                    "part": {
                        "type": "step-finish",
                        "reason": "tool-calls",
                        "tokens": {
                            "input": 500,
                            "output": 100,
                            "reasoning": 0,
                            "cache": {"read": 200, "write": 0},
                        },
                    },
                }
            )
            time.sleep(0.2)
        return 0

    if scenario == "single_step_stop":
        _emit({"type": "step_start", "sessionID": "ses_4", "part": {"type": "step-start"}})
        _emit(
            {
                "type": "text",
                "sessionID": "ses_4",
                "part": {"type": "text", "text": "Done in one."},
            }
        )
        _emit(
            {
                "type": "step_finish",
                "sessionID": "ses_4",
                "part": {
                    "type": "step-finish",
                    "reason": "stop",
                    "tokens": {
                        "input": 20,
                        "output": 5,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
            }
        )
        return 0

    if scenario == "hang_with_grandchild":
        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pid_file = os.environ.get("FAKE_OC_GRANDCHILD_PID_FILE")
        if pid_file:
            with open(pid_file, "w", encoding="utf-8") as handle:
                handle.write(str(grandchild.pid))
        _emit({"type": "step_start", "sessionID": "ses_hang", "part": {"type": "step-start"}})
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
        return 0

    if scenario == "parent_exits_grandchild_holds_pipe":
        grandchild = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "time.sleep(60)"
                ),
            ]
        )
        pid_file = os.environ.get("FAKE_OC_GRANDCHILD_PID_FILE")
        if pid_file:
            with open(pid_file, "w", encoding="utf-8") as handle:
                handle.write(str(grandchild.pid))
        _emit({"type": "step_start", "sessionID": "ses_pipe", "part": {"type": "step-start"}})
        time.sleep(60)
        return 0

    if scenario == "error_event":
        secret = os.environ.get("FAKE_OC_LEAK_SECRET", "")
        _emit(
            {
                "type": "error",
                "sessionID": "ses_err",
                "error": {
                    "type": "provider.no-route",
                    "message": f"Model unavailable with key {secret}",
                },
            }
        )
        return 1

    if scenario == "transient_error_then_ok":
        _emit(
            {
                "type": "error",
                "sessionID": "ses_retry",
                "error": {"type": "transport", "message": "temporary 502"},
            }
        )
        _emit({"type": "step_start", "sessionID": "ses_retry", "part": {"type": "step-start"}})
        _emit(
            {
                "type": "text",
                "sessionID": "ses_retry",
                "part": {"type": "text", "text": "Recovered"},
            }
        )
        _emit(
            {
                "type": "step_finish",
                "sessionID": "ses_retry",
                "part": {
                    "type": "step-finish",
                    "reason": "stop",
                    "tokens": {
                        "input": 10,
                        "output": 4,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
            }
        )
        return 0

    if scenario == "stderr_fail":
        secret = os.environ.get("FAKE_OC_LEAK_SECRET", "")
        sys.stderr.write(f"fatal crash leaking {secret}\\n")
        sys.stderr.flush()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _install_fake_opencode(tmp_path: Path, name: str = "fake-opencode") -> Path:
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\n{_FAKE_OPENCODE_SOURCE}", encoding="utf-8")
    script.chmod(0o755)
    _reset_opencode_availability_cache()
    return script


class _GatewayProvider(MockLLMProvider):
    def __init__(self, binding: ExternalGatewayBinding) -> None:
        super().__init__(responses=[])
        self._binding = binding

    def external_gateway_binding(self, model: str | None = None) -> ExternalGatewayBinding:
        if model:
            return ExternalGatewayBinding(
                api=self._binding.api,
                base_url=self._binding.base_url,
                model_id=model,
                api_key=self._binding.api_key,
                api_key_env=self._binding.api_key_env,
                is_custom_endpoint=self._binding.is_custom_endpoint,
            )
        return self._binding


def _new_store_and_run(
    tmp_path: Path,
    *,
    max_tokens: int | None = None,
    max_llm_calls: int | None = None,
) -> tuple[RunStore, str]:
    store = RunStore(tmp_path / "olympus.db")
    policy_kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        policy_kwargs["max_total_tokens"] = max_tokens
    if max_llm_calls is not None:
        policy_kwargs["max_llm_calls"] = max_llm_calls
    policy = BudgetPolicy(**policy_kwargs)
    run_id = store.create_run(
        title="test goal",
        mode=RunMode.SIMULATION,
        budget_policy=policy,
    )
    return store, run_id


def test_opencode_driver_streams_events_usage_receipt_and_cleans_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = _install_fake_opencode(tmp_path)
    capture_path = tmp_path / "capture.json"
    trace_path = tmp_path / "trace.jsonl"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("FAKE_OC_SCENARIO", "happy")
    monkeypatch.setenv("FAKE_OC_CAPTURE_PATH", str(capture_path))
    monkeypatch.setenv("OPENCODE_DB", str(tmp_path / "host.db"))
    monkeypatch.setenv("OPENCODE_CONFIG", str(tmp_path / "host.json"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "host-cfg"))
    monkeypatch.setenv("OPENCODE_PERMISSION", '{"*":"allow"}')
    monkeypatch.setenv("OPENCODE_DISABLE_PROJECT_CONFIG", "0")
    monkeypatch.setenv("OPENCODE_CONFIG_PROJECT_DISABLE", "0")
    monkeypatch.setenv("OPENCODE_DISABLE_CLAUDE_CODE", "0")

    store, run_id = _new_store_and_run(tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []
    handles: list[ActiveTurnHandle] = []

    driver = OpenCodeDriver(binary=str(fake_bin), store=store, run_id=run_id)
    outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="You are Hephaestus.",
            user_prompt="--implement feature safely",
            actor="hephaestus",
            task_id="T1",
            write_scope=["src/app.py"],
            trace_path=trace_path,
            model="opencode/hy4-flash-free",
        ),
        on_event=lambda kind, payload: events.append((kind, payload)),
        on_handle=handles.append,
    )

    assert outcome.completed is True
    assert outcome.stop_reason == "final_response"
    assert outcome.final_content == "Final answer OK"
    assert outcome.steps_taken == 2
    assert outcome.total_input_tokens == 150
    # OpenCode reports output excluding reasoning; Olympus output_tokens includes reasoning.
    assert outcome.total_output_tokens == 42
    assert outcome.total_cache_read_tokens == 70
    assert outcome.total_cache_write_tokens == 10
    assert outcome.total_reasoning_tokens == 7
    assert outcome.driver_name == "opencode"

    assert len(handles) == 1
    assert handles[0].supports_steer is False
    assert handles[0].steer("try harder", source="user") is False

    kinds = [kind for kind, _ in events]
    assert "agent.tool_started" in kinds
    assert "agent.tool_finished" in kinds
    assert "text_delta" in kinds
    assert kinds.count("llm") == 2
    assert kinds[-1] == "receipt"

    tool_finished = next(payload for kind, payload in events if kind == "agent.tool_finished")
    assert tool_finished["tool"] == "read"
    assert tool_finished["status"] == "ok"
    assert tool_finished["duration_ms"] == 45
    assert tool_finished["error_signature"] == ""

    ledger = store.get_run(run_id)["budget_ledger"]
    assert ledger.input_tokens == 150
    assert ledger.output_tokens == 42
    assert ledger.cache_read_tokens == 70
    assert ledger.cache_write_tokens == 10
    assert ledger.reasoning_tokens == 7
    assert ledger.total_tokens == 272
    assert ledger.llm_calls_used == 2

    receipt_events = [e for e in store.list_events(run_id) if e.event_type == "turn.receipt"]
    assert len(receipt_events) == 1
    assert receipt_events[0].payload["llm"]["calls"] == 2
    assert receipt_events[0].payload["llm"]["total_tokens"] == 272

    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    assert captured["cwd"] == str(workspace.resolve())
    assert captured["pwd"] == str(workspace.resolve())
    assert captured["opencode_db"] is None
    assert captured["opencode_config"] is None
    assert captured["opencode_config_dir"] is None
    assert captured["opencode_permission"] is None
    assert "--title" in captured["argv"]
    assert "olympus:hephaestus" in captured["argv"]
    assert captured["argv"][-2:] == ["--", "--implement feature safely"]
    assert "--model" in captured["argv"]
    assert "opencode/hy4-flash-free" in captured["argv"]
    assert "provider" not in captured["config"]
    assert captured["config"]["agent"]["olympus_turn"]["prompt"] == "You are Hephaestus."
    assert captured["disable_claude"] == "1"
    assert captured["disable_project_config"] == "1"
    assert captured["config_project_disable"] == "1"
    assert not Path(captured["xdg_data"]).exists()

    trace_records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    spawn_record = next(r for r in trace_records if r["event"] == "opencode.spawn")
    assert spawn_record["argv"][-1].startswith("<prompt: ")


def test_opencode_driver_permissions_and_denied_tool_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = _install_fake_opencode(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    ro_perms = _build_agent_permissions(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="inspect",
            actor="argus",
            read_only=True,
        )
    )
    assert ro_perms["*"] == "deny"
    assert ro_perms["read"] == "allow"
    assert "edit" not in ro_perms
    assert "bash" not in ro_perms
    assert "shell" not in ro_perms

    no_tool_perms = _build_agent_permissions(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="answer",
            actor="prometheus",
            disable_tools=True,
        )
    )
    assert no_tool_perms == {"*": "deny"}

    scoped_perms = _build_agent_permissions(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="edit",
            actor="hephaestus",
            strict_scope=True,
            write_scope=["src/app.py", "src/pkg/"],
            forbidden_globs=["src/pkg/secret.py", "core/**"],
        )
    )
    edit_items = list(scoped_perms["edit"].items())
    assert edit_items[0] == ("*", "deny")
    assert scoped_perms["edit"]["src/app.py"] == "allow"
    assert scoped_perms["edit"]["src/pkg/**"] == "allow"
    assert scoped_perms["edit"]["src/pkg/secret.py"] == "deny"
    assert scoped_perms["edit"]["core/**"] == "deny"
    assert scoped_perms["external_directory"] == "deny"

    relaxed_perms = _build_agent_permissions(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="edit",
            actor="hephaestus",
            strict_scope=False,
            forbidden_globs=["core/**"],
        )
    )
    assert list(relaxed_perms["edit"].items())[0] == ("*", "allow")
    assert relaxed_perms["edit"]["core/**"] == "deny"

    overlap_perms = _build_agent_permissions(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="edit",
            actor="hephaestus",
            strict_scope=True,
            write_scope=[
                "src/secret.py",
                "src/**",
                "out-[ab].txt",
                "v[1-3].py",
                "allow-[!z].txt",
                "neg-*.txt",
            ],
            forbidden_globs=[
                "src/secret.py",
                "deny-[xy].txt",
                "neg-[!z].txt",
                "report[a/]draft.txt",
            ],
        )
    )
    overlap_keys = list(overlap_perms["edit"].keys())
    assert overlap_keys.index("src/secret.py") > overlap_keys.index("src/**")
    assert overlap_perms["edit"]["src/secret.py"] == "deny"
    assert overlap_perms["edit"]["out-a.txt"] == "allow"
    assert overlap_perms["edit"]["out-b.txt"] == "allow"
    assert "out-[ab].txt" not in overlap_perms["edit"]
    assert overlap_perms["edit"]["v1.py"] == "allow"
    assert overlap_perms["edit"]["v2.py"] == "allow"
    assert overlap_perms["edit"]["v3.py"] == "allow"
    # Negated write_scope: allow-?.txt -> allow, then allow-z.txt -> deny
    assert overlap_perms["edit"]["allow-?.txt"] == "allow"
    assert overlap_perms["edit"]["allow-z.txt"] == "deny"
    assert overlap_keys.index("allow-z.txt") > overlap_keys.index("allow-?.txt")
    # Negated forbidden_globs with neg-*.txt write_scope: neg-?.txt -> deny, neg-z.txt -> allow
    assert overlap_perms["edit"]["neg-?.txt"] == "deny"
    assert overlap_perms["edit"]["neg-z.txt"] == "allow"
    assert overlap_keys.index("neg-z.txt") > overlap_keys.index("neg-?.txt")
    assert overlap_perms["edit"]["deny-x.txt"] == "deny"
    assert overlap_perms["edit"]["deny-y.txt"] == "deny"
    assert overlap_perms["edit"]["reportadraft.txt"] == "deny"
    assert overlap_perms["edit"]["report/draft.txt"] == "deny"

    # Unrepresentable globs fail fast instead of widening or mis-denying
    for bad_write_scope, bad_forbidden in (
        (["huge-[a-z][a-z].txt"], []),
        (["two-[!a][!b].txt"], []),
        (["*/neg-[!a].txt"], []),
        (["src/**"], ["*/neg-[!a].txt"]),
        (["report-[?].txt"], []),
        (["report-[*].txt"], []),
        (["range-[--@].txt"], []),
        (["src/**"], ["report-[?].txt"]),
    ):
        with pytest.raises(ValueError, match="OpenCode declarative permission rules"):
            _build_agent_permissions(
                DriverTurnRequest(
                    workspace_path=workspace,
                    env_root=workspace,
                    system_prompt="",
                    user_prompt="edit",
                    actor="hephaestus",
                    strict_scope=True,
                    write_scope=bad_write_scope,
                    forbidden_globs=bad_forbidden,
                )
            )

    monkeypatch.setenv("FAKE_OC_SCENARIO", "permission_denied")
    events: list[tuple[str, dict[str, Any]]] = []
    driver = OpenCodeDriver(binary=str(fake_bin))
    outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="try edit",
            actor="hephaestus",
            write_scope=["allowed/**"],
            model="opencode/fake-model",
        ),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )
    assert outcome.completed is True
    assert outcome.receipt["tools"] == [{"tool": "edit", "status": "denied", "duration_ms": 0}]
    finished = next(payload for kind, payload in events if kind == "agent.tool_finished")
    assert finished["status"] == "error"
    assert finished["error_signature"] != ""


def test_opencode_driver_synthesizes_olympus_gateway_without_leaking_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = _install_fake_opencode(tmp_path)
    capture_path = tmp_path / "capture.json"
    trace_path = tmp_path / "trace.jsonl"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret_value = "sk-super-secret-olympus-key-99999"

    monkeypatch.setenv("FAKE_OC_SCENARIO", "happy")
    monkeypatch.setenv("FAKE_OC_CAPTURE_PATH", str(capture_path))

    provider = _GatewayProvider(
        ExternalGatewayBinding(
            api="anthropic-messages",
            base_url="https://proxy.example.invalid/v1",
            model_id="claude-sonnet-4-6",
            api_key=secret_value,
        )
    )
    driver = OpenCodeDriver(binary=str(fake_bin), provider=provider, context_window=64000)
    outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="Be concise.",
            user_prompt="Hello",
            actor="hephaestus",
            trace_path=trace_path,
        )
    )
    assert outcome.completed is True

    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    raw_config = json.dumps(captured["config"])
    assert secret_value not in raw_config
    assert secret_value not in trace_path.read_text(encoding="utf-8")
    assert captured["gateway_key_present"] is True
    gw = captured["config"]["provider"]["olympus-gateway"]
    assert gw["npm"] == "@ai-sdk/anthropic"
    assert gw["options"]["baseURL"] == "https://proxy.example.invalid/v1"
    assert gw["options"]["apiKey"] == "{env:OLYMPUS_OPENCODE_GATEWAY_KEY}"
    assert gw["models"]["claude-sonnet-4-6"]["limit"]["context"] == 64000
    assert "olympus-gateway/claude-sonnet-4-6" in captured["argv"]

    # Bare opencode without any Olympus ExternalGatewayBinding must fail fast (Decision 3A)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    unbound_driver = OpenCodeDriver(binary=str(fake_bin))
    with pytest.raises(OpenCodeNotAvailableError, match="ExternalGatewayBinding"):
        unbound_driver.run_turn(
            DriverTurnRequest(
                workspace_path=workspace,
                env_root=workspace,
                system_prompt="",
                user_prompt="hello",
                actor="hephaestus",
            )
        )


def test_opencode_driver_enforces_budget_and_max_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = _install_fake_opencode(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # Budget exhaustion stops on step 1 (500+100+200 = 800 > 500)
    store, run_id = _new_store_and_run(tmp_path, max_tokens=500)
    monkeypatch.setenv("FAKE_OC_SCENARIO", "multi_step_loop")
    driver = OpenCodeDriver(
        binary=str(fake_bin),
        store=store,
        run_id=run_id,
        term_grace_seconds=0.3,
        cancel_grace_seconds=0.5,
    )
    budget_outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="loop",
            actor="hephaestus",
            model="opencode/fake-model",
        )
    )
    assert budget_outcome.completed is False
    assert budget_outcome.stop_reason.startswith("budget_exhausted:")
    assert budget_outcome.steps_taken == 1

    # max_llm_calls=1 cancels on step 1 when reason="tool-calls" and wires agent.steps=1
    call_cap_dir = tmp_path / "callcap"
    call_cap_dir.mkdir(exist_ok=True)
    call_cap_store, call_cap_run = _new_store_and_run(call_cap_dir, max_llm_calls=1)
    call_capture = tmp_path / "call_cap_capture.json"
    monkeypatch.setenv("FAKE_OC_SCENARIO", "multi_step_loop")
    monkeypatch.setenv("FAKE_OC_CAPTURE_PATH", str(call_capture))
    call_cap_driver = OpenCodeDriver(
        binary=str(fake_bin),
        store=call_cap_store,
        run_id=call_cap_run,
        term_grace_seconds=0.3,
        cancel_grace_seconds=0.5,
    )
    call_cap_outcome = call_cap_driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="loop",
            actor="hephaestus",
            model="opencode/fake-model",
        )
    )
    assert call_cap_outcome.completed is False
    assert call_cap_outcome.stop_reason.startswith("budget_exhausted:")
    assert call_cap_outcome.steps_taken == 1
    assert call_cap_store.get_run(call_cap_run)["budget_ledger"].llm_calls_used == 1
    cap_captured = json.loads(call_capture.read_text(encoding="utf-8"))
    assert cap_captured["config"]["agent"]["olympus_turn"]["steps"] == 1
    monkeypatch.delenv("FAKE_OC_CAPTURE_PATH", raising=False)

    # A follow-up turn when budget is already exhausted fails preflight without spawning
    preflight_outcome = call_cap_driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="should not spawn",
            actor="hephaestus",
            model="opencode/fake-model",
        )
    )
    assert preflight_outcome.completed is False
    assert preflight_outcome.stop_reason.startswith("budget_exhausted:")
    assert preflight_outcome.steps_taken == 0
    assert call_cap_store.get_run(call_cap_run)["budget_ledger"].llm_calls_reserved == 0

    # Concurrent turns sharing 1 remaining call reserve atomically (at most 1 spawns)
    race_dir = tmp_path / "racecap"
    race_dir.mkdir(exist_ok=True)
    race_store, race_run = _new_store_and_run(race_dir, max_llm_calls=1)
    monkeypatch.setenv("FAKE_OC_SCENARIO", "single_step_stop")
    barrier = Barrier(2)
    race_outcomes: list[Any] = []

    def _run_racer(actor_name: str) -> None:
        racer = OpenCodeDriver(binary=str(fake_bin), store=race_store, run_id=race_run)
        barrier.wait(timeout=5.0)
        race_outcomes.append(
            racer.run_turn(
                DriverTurnRequest(
                    workspace_path=workspace,
                    env_root=workspace,
                    system_prompt="",
                    user_prompt="race",
                    actor=actor_name,
                    model="opencode/fake-model",
                )
            )
        )

    t1 = Thread(target=_run_racer, args=("hephaestus-1",))
    t2 = Thread(target=_run_racer, args=("hephaestus-2",))
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)
    assert len(race_outcomes) == 2
    assert sum(1 for o in race_outcomes if o.completed) == 1
    assert sum(1 for o in race_outcomes if o.stop_reason.startswith("budget_exhausted:")) == 1
    race_ledger = race_store.get_run(race_run)["budget_ledger"]
    assert race_ledger.llm_calls_used == 1
    assert race_ledger.llm_calls_reserved == 0

    # Concurrent multi-step turns on max_llm_calls=2 never exceed 2 total calls
    race2_dir = tmp_path / "racecap2"
    race2_dir.mkdir(exist_ok=True)
    race2_store, race2_run = _new_store_and_run(race2_dir, max_llm_calls=2)
    monkeypatch.setenv("FAKE_OC_SCENARIO", "multi_step_loop")
    barrier2 = Barrier(2)
    race2_outcomes: list[Any] = []

    def _run_racer2(actor_name: str) -> None:
        racer = OpenCodeDriver(
            binary=str(fake_bin),
            store=race2_store,
            run_id=race2_run,
            term_grace_seconds=0.3,
            cancel_grace_seconds=0.5,
        )
        barrier2.wait(timeout=5.0)
        race2_outcomes.append(
            racer.run_turn(
                DriverTurnRequest(
                    workspace_path=workspace,
                    env_root=workspace,
                    system_prompt="",
                    user_prompt="race2",
                    actor=actor_name,
                    model="opencode/fake-model",
                )
            )
        )

    r2_t1 = Thread(target=_run_racer2, args=("hephaestus-a",))
    r2_t2 = Thread(target=_run_racer2, args=("hephaestus-b",))
    r2_t1.start()
    r2_t2.start()
    r2_t1.join(timeout=10.0)
    r2_t2.join(timeout=10.0)
    assert len(race2_outcomes) == 2
    race2_ledger = race2_store.get_run(race2_run)["budget_ledger"]
    assert race2_ledger.llm_calls_used <= 2
    assert race2_ledger.llm_calls_reserved == 0

    # Crashed turn releases its reserved slot cleanly
    crash_dir = tmp_path / "crashcap"
    crash_dir.mkdir(exist_ok=True)
    crash_store, crash_run = _new_store_and_run(crash_dir, max_llm_calls=1)
    monkeypatch.setenv("FAKE_OC_SCENARIO", "stderr_fail")
    crash_driver = OpenCodeDriver(binary=str(fake_bin), store=crash_store, run_id=crash_run)
    crash_outcome = crash_driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="crash",
            actor="hephaestus",
            model="opencode/fake-model",
        )
    )
    assert crash_outcome.completed is False
    crash_ledger = crash_store.get_run(crash_run)["budget_ledger"]
    assert crash_ledger.llm_calls_used == 0
    assert crash_ledger.llm_calls_reserved == 0

    # max_steps=1 with tool-calls terminates after step 1
    monkeypatch.setenv("FAKE_OC_SCENARIO", "multi_step_loop")
    driver_no_store = OpenCodeDriver(
        binary=str(fake_bin), term_grace_seconds=0.3, cancel_grace_seconds=0.5
    )
    step_outcome = driver_no_store.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="loop",
            actor="hephaestus",
            max_steps=1,
            model="opencode/fake-model",
        )
    )
    assert step_outcome.completed is False
    assert step_outcome.stop_reason == "max_steps_exceeded"
    assert step_outcome.steps_taken == 1

    # max_steps=1 with reason="stop" completes cleanly
    monkeypatch.setenv("FAKE_OC_SCENARIO", "single_step_stop")
    ok_outcome = driver_no_store.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="one shot",
            actor="hephaestus",
            max_steps=1,
            model="opencode/fake-model",
        )
    )
    assert ok_outcome.completed is True
    assert ok_outcome.stop_reason == "final_response"
    assert ok_outcome.final_content == "Done in one."


@pytest.mark.parametrize(
    "scenario",
    ["hang_with_grandchild", "parent_exits_grandchild_holds_pipe"],
)
def test_opencode_driver_cancel_reclaims_process_group_with_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    fake_bin = _install_fake_opencode(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setenv("FAKE_OC_SCENARIO", scenario)
    monkeypatch.setenv("FAKE_OC_GRANDCHILD_PID_FILE", str(pid_file))

    driver = OpenCodeDriver(
        binary=str(fake_bin), term_grace_seconds=0.2, cancel_grace_seconds=0.5
    )
    handle_ready = Event()
    captured_handle: list[ActiveTurnHandle] = []

    def on_handle(handle: ActiveTurnHandle) -> None:
        captured_handle.append(handle)
        handle_ready.set()

    def cancel_soon() -> None:
        handle_ready.wait(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        captured_handle[0].cancel(reason="user_abort")

    started_at = time.monotonic()
    canceller = Thread(target=cancel_soon, daemon=True)
    canceller.start()
    outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="hang",
            actor="hephaestus",
            model="opencode/fake-model",
        ),
        on_handle=on_handle,
    )
    canceller.join(timeout=5.0)
    elapsed = time.monotonic() - started_at

    assert outcome.completed is False
    assert outcome.stop_reason == "cancelled"
    assert elapsed < 3.0
    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    deadline = time.monotonic() + 2.0
    alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            alive = False
            break
    assert not alive


def test_opencode_driver_errors_redaction_and_transient_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = _install_fake_opencode(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = "sk-leak-check-abcdef123456"
    monkeypatch.setenv("FAKE_OC_LEAK_SECRET", secret)
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    driver = OpenCodeDriver(binary=str(fake_bin))

    monkeypatch.setenv("FAKE_OC_SCENARIO", "error_event")
    err_outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="fail",
            actor="hephaestus",
            model="opencode/fake-model",
        )
    )
    assert err_outcome.completed is False
    assert err_outcome.stop_reason.startswith("opencode_error:")
    assert secret not in err_outcome.stop_reason
    assert "***" in err_outcome.stop_reason

    monkeypatch.setenv("FAKE_OC_SCENARIO", "stderr_fail")
    stderr_outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="crash",
            actor="hephaestus",
            model="opencode/fake-model",
        )
    )
    assert stderr_outcome.completed is False
    assert stderr_outcome.stop_reason.startswith("opencode_error:")
    assert secret not in stderr_outcome.stop_reason
    assert "***" in stderr_outcome.stop_reason

    monkeypatch.setenv("FAKE_OC_SCENARIO", "transient_error_then_ok")
    ok_outcome = driver.run_turn(
        DriverTurnRequest(
            workspace_path=workspace,
            env_root=workspace,
            system_prompt="",
            user_prompt="recover",
            actor="hephaestus",
            model="opencode/fake-model",
        )
    )
    assert ok_outcome.completed is True
    assert ok_outcome.final_content == "Recovered"


def test_opencode_availability_registry_and_auto_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_opencode_availability_cache()
    assert opencode_available(binary="/nonexistent/path/to/opencode") is False
    with pytest.raises(OpenCodeNotAvailableError):
        OpenCodeDriver(binary="/nonexistent/path/to/opencode")

    fake_bin = _install_fake_opencode(tmp_path, name="opencode-bad-help")
    monkeypatch.setenv("FAKE_OC_NO_JSON_HELP", "1")
    assert opencode_available(binary=str(fake_bin)) is False

    monkeypatch.delenv("FAKE_OC_NO_JSON_HELP", raising=False)
    _reset_opencode_availability_cache()
    assert opencode_available(binary=str(fake_bin)) is True

    monkeypatch.setenv("OLYMPUS_OPENCODE_BIN", str(fake_bin))
    factories = default_driver_factories(
        store=None,
        run_id=None,
        provider=None,
        execution_drivers=ExecutionDriversConfig(),
    )
    assert "opencode" in factories
    registry = DriverRegistry(factories=factories)
    driver, model = registry.resolve_external("opencode@opencode/hy4-flash-free#high")
    assert isinstance(driver, OpenCodeDriver)
    assert model == "opencode/hy4-flash-free#high"

    # auto never routes to opencode (Decision 3A)
    monkeypatch.setattr("olympus.harness.pi.pi_available", lambda **_: False)
    assert resolve_driver_spec("auto", provider=MockLLMProvider(responses=[])) == "native"

    assert _safe_argv(["opencode", "run", "--", "secret prompt text"]) == [
        "run",
        "--",
        "<prompt: 18 chars>",
    ]
