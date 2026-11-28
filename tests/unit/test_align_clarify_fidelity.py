"""Unit tests for clarification fidelity, extra-line preservation, and post-clarify resynthesis."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from olympus.core.contracts import (
    AcceptanceCriterion,
    AmbiguityItem,
    DeliverableKind,
    EarsPattern,
    GoalSpec,
    RequirementItem,
    RunStatus,
    SpecStatus,
)
from olympus.core.control import RunControl
from olympus.core.store import ConcurrencyConflictError, RunStore
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
from olympus.session.contracts import GatePolicyName, PauseKind, SessionStatus, StepMode
from olympus.session.router import parse_reply
from olympus.session.runner import SessionConfig, SessionRunner
from olympus.stages.align import AlignStage


def _init_git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _draft_spec(
    *,
    spec_id: str = "spec-blacknote",
    ambiguities: list[AmbiguityItem] | None = None,
) -> GoalSpec:
    return GoalSpec(
        spec_id=spec_id,
        goal="构建面向 AI 工程师的图文分享与讨论社区 Web 应用",
        deliverable_kind=DeliverableKind.WEB_APP,
        observable_surfaces=["Web 图文信息流", "笔记详情与评论页"],
        actors=["AI 工程师", "管理员"],
        test_seams=["HTTP REST API 与 Web 信息流页面"],
        requirements=[
            RequirementItem(
                id="R1",
                actor="AI 工程师",
                description=(
                    "支持浏览与发布图文笔记，并按若干话题标签（如 #AI新闻 #工具 #产品）筛选。"
                ),
                ears_criteria=[
                    AcceptanceCriterion(
                        id="AC-1",
                        pattern=EarsPattern.EVENT_DRIVEN,
                        condition="AI 工程师提交图文笔记",
                        outcome="在 1 秒内持久化并展示在信息流中",
                    )
                ],
            )
        ],
        constraints=["使用服务端数据库持久化"],
        assumptions=["允许匿名游客直接浏览信息流"],
        out_of_scope=["关注关系图谱与私有私信"],
        open_ambiguities=ambiguities
        if ambiguities is not None
        else [
            AmbiguityItem(
                id="A1-auth-anon",
                question="未登录用户是否可浏览？",
                impact="决定信息流访问边界",
                options=["公开浏览+登录发帖", "强制登录后访问"],
                default_assumption="公开浏览+登录发帖",
                blocking=True,
            ),
            AmbiguityItem(
                id="A2-media-source",
                question="笔记配图如何提供？",
                impact="决定图片上传接口",
                options=["本地上传或图片 URL", "仅图片 URL"],
                default_assumption="本地上传或图片 URL",
                blocking=True,
            ),
        ],
        status=SpecStatus.AWAITING_HUMAN,
    )


def test_router_fast_path_preserves_preamble_and_trailing_lines_with_explicit_ids() -> None:
    spec = _draft_spec()
    snapshot = {"spec": spec, "plan": None, "status": RunStatus.AWAITING_HUMAN}

    def _never_called(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        raise AssertionError("deterministic fast-path should not call LLM")

    reply = (
        "产品核心边界如下：\n"
        "1. 浏览器 Web 图文信息流，而非仅 API/CLI\n"
        "2. 话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类并支持自定义标签\n"
        "A1-auth-anon: 公开浏览+登录发帖，内置邮箱+密码账号体系并预置管理员账号\n"
        "A2-media-source: 支持本地上传与图片 URL\n"
        "补充：所有数据持久化到服务端数据库，重启不丢数据。"
    )
    intent = parse_reply(
        provider=MockLLMProvider(responder=_never_called),
        message=reply,
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert intent.action == "clarify"
    assert intent.answers == {
        "A1-auth-anon": "公开浏览+登录发帖，内置邮箱+密码账号体系并预置管理员账号",
        "A2-media-source": "支持本地上传与图片 URL",
    }
    assert "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类并支持自定义标签" in intent.feedback
    assert "重启不丢数据" in intent.feedback
    assert "A1-auth-anon" not in intent.feedback
    assert "A2-media-source" not in intent.feedback


def test_router_fast_path_default_delegation_filters_only_delegation_line() -> None:
    spec = _draft_spec()
    snapshot = {"spec": spec, "plan": None, "status": RunStatus.AWAITING_HUMAN}

    def _never_called(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        raise AssertionError("deterministic fast-path should not call LLM")

    pure_defaults = parse_reply(
        provider=MockLLMProvider(responder=_never_called),
        message="Use your default assumptions and proceed.",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert pure_defaults.action == "clarify"
    assert pure_defaults.answers == {
        "A1-auth-anon": "公开浏览+登录发帖",
        "A2-media-source": "本地上传或图片 URL",
    }
    assert pure_defaults.feedback == ""

    defaults_plus_rule = parse_reply(
        provider=MockLLMProvider(responder=_never_called),
        message=(
            "Use your default assumptions and proceed.\n"
            "话题必须固定包含 AI 新闻 / AI 工具 / AI 产品 三类。"
        ),
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert defaults_plus_rule.action == "clarify"
    assert defaults_plus_rule.feedback == "话题必须固定包含 AI 新闻 / AI 工具 / AI 产品 三类。"


def test_router_sanitizer_attaches_extra_lines_on_partial_explicit_answers() -> None:
    spec = _draft_spec()
    snapshot = {"spec": spec, "plan": None, "status": RunStatus.AWAITING_HUMAN}
    provider = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(
            content=json.dumps(
                {
                    "action": "clarify",
                    "answers": {"A1-auth-anon": "公开浏览+登录发帖"},
                    "resolutions": {},
                    "feedback": "",
                    "message": "Recorded A1.",
                }
            )
        )
    )
    intent = parse_reply(
        provider=provider,
        message="话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类。\nA1-auth-anon: 公开浏览+登录发帖",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert intent.action == "clarify"
    assert intent.answers == {"A1-auth-anon": "公开浏览+登录发帖"}
    assert intent.feedback == "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类。"


def test_align_resynthesize_fuses_clarifications_and_locks_deterministic_fields(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(title="blacknote", workflow_name="software_delivery")
    draft = _draft_spec()
    store.save_spec(run_id, draft, actor="align")
    clarified = RunControl(store).clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "强制登录后访问，内置邮箱+密码与预置管理员账号",
            "A2-media-source": "本地上传或图片 URL",
        },
        actor="human",
    )

    seen_prompts: list[str] = []

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        seen_prompts.append(messages[1].content)
        mutated = clarified.model_copy(
            update={
                "spec_id": "hijacked-id",
                "version": 1,
                "status": SpecStatus.CONFIRMED,
                "requirements": [
                    RequirementItem(
                        id="R1",
                        actor="AI 工程师",
                        description=(
                            "平台固定提供 AI 新闻、AI 工具、AI 产品 三类一级话题，"
                            "并支持自定义标签。"
                        ),
                        ears_criteria=[
                            AcceptanceCriterion(
                                id="AC-1",
                                pattern=EarsPattern.UBIQUITOUS,
                                outcome=(
                                    "在信息流顶部展示 AI 新闻、AI 工具、AI 产品 "
                                    "三个固定话题筛选入口"
                                ),
                            )
                        ],
                    )
                ],
                "constraints": [
                    "使用服务端数据库持久化",
                    "强制登录后访问，内置邮箱+密码与预置管理员账号",
                    "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类并支持自定义标签",
                ],
                "open_ambiguities": [
                    AmbiguityItem(
                        id="A99-smuggled",
                        question="Should never survive deterministic lock",
                        impact="none",
                        default_assumption="no",
                        blocking=True,
                    )
                ],
            }
        )
        return LLMResponse(content=mutated.model_dump_json())

    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=_responder),
        workspace_root=tmp_path,
    )
    updated = stage.resynthesize_after_clarification(
        run_id=run_id,
        raw_goal="构建面向 AI 工程师的图文分享社区",
        extra_feedback="话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类并支持自定义标签",
    )

    assert len(seen_prompts) == 1
    assert "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类并支持自定义标签" in seen_prompts[0]
    assert "强制登录后访问" in seen_prompts[0]
    assert updated.spec_id == clarified.spec_id
    assert updated.version == clarified.version + 1
    assert updated.status == SpecStatus.AWAITING_HUMAN
    assert updated.open_ambiguities == clarified.open_ambiguities
    assert updated.unresolved_blockers() == []
    assert updated.content_hash != clarified.content_hash
    assert any("AI 新闻" in req.description for req in updated.requirements)

    resynth_events = [
        ev for ev in store.list_events(run_id) if ev.event_type == "spec.resynthesized"
    ]
    assert len(resynth_events) == 1
    assert resynth_events[0].payload["previous_spec_hash"] == clarified.content_hash
    assert resynth_events[0].payload["spec_hash"] == updated.content_hash
    assert resynth_events[0].payload["had_extra_feedback"] is True


def _design_json(spec_id: str, spec_hash: str) -> str:
    return json.dumps(
        {
            "architecture_summary": "FastAPI + SQLite + Web UI covering fixed topics.",
            "spec_id": spec_id,
            "spec_hash": spec_hash,
            "shared_invariants": [
                {
                    "id": "INV-TOPICS",
                    "category": "validation",
                    "rule": "Fixed AI News / AI Tools / AI Products topics are always persisted.",
                    "enforcement_hint": "seed fixed topic enum",
                    "applies_to": ["M1"],
                }
            ],
            "shared_contract_paths": [],
            "modules": [
                {
                    "id": "M1",
                    "name": "NotesService",
                    "responsibility": "Manage notes and fixed AI topics.",
                    "requirement_refs": ["R1"],
                    "target_paths": ["app.py"],
                }
            ],
            "data_models": [],
            "interfaces": [],
            "execution_strategy": "serial_handoff",
            "strategy_rationale": "Single compact module keeps delivery atomic.",
        }
    )


def _plan_json(spec_id: str, spec_hash: str, design_id: str, design_hash: str) -> str:
    return json.dumps(
        {
            "plan_id": "plan-blacknote",
            "spec_id": spec_id,
            "spec_hash": spec_hash,
            "design_id": design_id,
            "design_hash": design_hash,
            "version": 1,
            "tasks": [
                {
                    "task_id": "T1",
                    "title": "Build fixed-topic notes app",
                    "objective": "Implement web feed with fixed AI topics.",
                    "requirement_refs": ["R1"],
                    "design_refs": ["M1"],
                    "write_scope": ["app.py"],
                    "acceptance_checks": [
                        {
                            "id": "C1",
                            "kind": "command",
                            "description": "Verify fixed topic feed",
                            "command": "python3 -c 'print(1)'",
                        }
                    ],
                }
            ],
        }
    )


def test_session_runner_resynthesizes_and_passes_fused_spec_to_design(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    draft = _draft_spec()
    fused_spec = draft.model_copy(
        update={
            "requirements": [
                RequirementItem(
                    id="R1",
                    actor="AI 工程师",
                    description=(
                        "平台固定包含 AI 新闻 / AI 工具 / AI 产品 三类一级话题"
                        "并支持自定义标签。"
                    ),
                    ears_criteria=[
                        AcceptanceCriterion(
                            id="AC-1",
                            pattern=EarsPattern.UBIQUITOUS,
                            outcome="固定提供 AI 新闻 / AI 工具 / AI 产品 三类话题筛选",
                        )
                    ],
                )
            ]
        }
    )

    align_calls: list[str] = []
    design_prompts: list[str] = []

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system = messages[0].content
        user = messages[1].content
        if "Requirements Alignment Architect" in system:
            align_calls.append(user)
            if len(align_calls) == 1:
                return LLMResponse(content=draft.model_dump_json())
            return LLMResponse(content=fused_spec.model_dump_json())
        if "White-Box Solution Architect" in system:
            design_prompts.append(user)
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            assert spec is not None
            return LLMResponse(content=_design_json(spec.spec_id, spec.content_hash))
        if "Task Planner" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            design = snap["design"]
            assert spec is not None and design is not None
            return LLMResponse(
                content=_plan_json(
                    spec.spec_id, spec.content_hash, design.design_id, design.content_hash
                )
            )
        raise AssertionError(f"Unexpected prompt: {system[:80]}")

    provider = MockLLMProvider(responder=_responder)
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: provider,
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    run_id = started.run_id
    assert started.status == SessionStatus.NEEDS_INPUT
    assert started.pause_kind == PauseKind.CLARIFY

    replied = runner.reply(
        run_id=run_id,
        message=(
            "2. 话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类并支持自定义标签\n"
            "A1-auth-anon: 公开浏览+登录发帖\n"
            "A2-media-source: 本地上传或图片 URL"
        ),
    )
    assert replied.status == SessionStatus.NEEDS_INPUT
    assert replied.pause_kind == PauseKind.PLAN_HANDOFF
    assert len(align_calls) == 2
    assert "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类并支持自定义标签" in align_calls[1]
    assert len(design_prompts) == 1
    assert "AI 新闻 / AI 工具 / AI 产品" in design_prompts[0]

    store = RunStore(state_dir / "olympus.db")
    event_types = [ev.event_type for ev in store.list_events(run_id)]
    assert "spec.clarification_note" in event_types
    assert "spec.resynthesized" in event_types


def test_session_runner_skips_resynthesis_on_pure_default_acceptance(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    draft = _draft_spec()
    align_calls = 0

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal align_calls
        del tools
        system = messages[0].content
        if "Requirements Alignment Architect" in system:
            align_calls += 1
            return LLMResponse(content=draft.model_dump_json())
        if "White-Box Solution Architect" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            assert spec is not None
            return LLMResponse(content=_design_json(spec.spec_id, spec.content_hash))
        if "Task Planner" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            design = snap["design"]
            assert spec is not None and design is not None
            return LLMResponse(
                content=_plan_json(
                    spec.spec_id, spec.content_hash, design.design_id, design.content_hash
                )
            )
        raise AssertionError(f"Unexpected prompt: {system[:80]}")

    provider = MockLLMProvider(responder=_responder)
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: provider,
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    run_id = started.run_id
    replied = runner.reply(run_id=run_id, message="Use your default assumptions and proceed.")
    assert replied.status == SessionStatus.NEEDS_INPUT
    assert replied.pause_kind == PauseKind.PLAN_HANDOFF
    assert align_calls == 1

    store = RunStore(state_dir / "olympus.db")
    event_types = [ev.event_type for ev in store.list_events(run_id)]
    assert "spec.resynthesized" not in event_types


def test_session_runner_multi_turn_clarification_retains_turn1_extra_feedback(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    draft = _draft_spec()
    fused = draft.model_copy(
        update={
            "requirements": [
                RequirementItem(
                    id="R1",
                    actor="AI 工程师",
                    description=(
                        "平台固定包含 AI 新闻 / AI 工具 / AI 产品 三类一级话题并支持自定义标签。"
                    ),
                    ears_criteria=[
                        AcceptanceCriterion(
                            id="AC-1",
                            pattern=EarsPattern.UBIQUITOUS,
                            outcome="固定展示 AI 新闻 / AI 工具 / AI 产品 三个话题入口",
                        )
                    ],
                )
            ]
        }
    )
    align_calls: list[str] = []

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system = messages[0].content
        user = messages[1].content
        if "Session Reply Router" in system:
            return LLMResponse(
                content=json.dumps(
                    {
                        "action": "clarify",
                        "answers": {"A1-auth-anon": "公开浏览+登录发帖"},
                        "resolutions": {},
                        "feedback": "",
                        "message": "Recorded A1.",
                    }
                )
            )
        if "Requirements Alignment Architect" in system:
            align_calls.append(user)
            return LLMResponse(
                content=draft.model_dump_json()
                if len(align_calls) == 1
                else fused.model_dump_json()
            )
        if "White-Box Solution Architect" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            assert spec is not None
            return LLMResponse(content=_design_json(spec.spec_id, spec.content_hash))
        if "Task Planner" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            design = snap["design"]
            assert spec is not None and design is not None
            return LLMResponse(
                content=_plan_json(
                    spec.spec_id, spec.content_hash, design.design_id, design.content_hash
                )
            )
        raise AssertionError(f"Unexpected prompt: {system[:80]}")

    provider = MockLLMProvider(responder=_responder)
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: provider,
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    run_id = started.run_id

    turn1 = runner.reply(
        run_id=run_id,
        message="话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类。\nA1-auth-anon: 公开浏览+登录发帖",
    )
    assert turn1.status == SessionStatus.NEEDS_INPUT
    assert turn1.pause_kind == PauseKind.CLARIFY
    assert len(align_calls) == 1

    turn2 = runner.reply(run_id=run_id, message="A2-media-source: 本地上传或图片 URL")
    assert turn2.status == SessionStatus.NEEDS_INPUT
    assert turn2.pause_kind == PauseKind.PLAN_HANDOFF
    assert len(align_calls) == 2
    assert "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类。" in align_calls[1]


def test_router_preserves_same_line_constraints_with_default_delegation() -> None:
    spec = _draft_spec()
    snapshot = {"spec": spec, "plan": None, "status": RunStatus.AWAITING_HUMAN}

    def _never_called(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        raise AssertionError("deterministic fast-path should not call LLM")

    en_mixed = parse_reply(
        provider=MockLLMProvider(responder=_never_called),
        message=(
            "Use your default assumptions, and topics must be fixed to "
            "AI News / AI Tools / AI Products."
        ),
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert en_mixed.action == "clarify"
    assert "AI News / AI Tools / AI Products" in en_mixed.feedback

    zh_mixed = parse_reply(
        provider=MockLLMProvider(responder=_never_called),
        message="按默认方案继续，但话题必须固定为 AI 新闻 / AI 工具 / AI 产品。",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert zh_mixed.action == "clarify"
    assert "AI 新闻 / AI 工具 / AI 产品" in zh_mixed.feedback

    zh_pure = parse_reply(
        provider=MockLLMProvider(responder=_never_called),
        message="按默认方案继续。",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert zh_pure.action == "clarify"
    assert zh_pure.feedback == ""


def test_router_sanitizer_preserves_unconsumed_prose_and_partial_llm_feedback() -> None:
    spec = _draft_spec()
    snapshot = {"spec": spec, "plan": None, "status": RunStatus.AWAITING_HUMAN}

    prose_provider = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(
            content=json.dumps(
                {
                    "action": "clarify",
                    "answers": {"A1-auth-anon": "公开浏览+登录发帖"},
                    "resolutions": {},
                    "feedback": "",
                    "message": "Parsed prose.",
                }
            )
        )
    )
    prose_intent = parse_reply(
        provider=prose_provider,
        message="公开浏览+登录发帖\n话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类。",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert prose_intent.action == "clarify"
    assert prose_intent.feedback == "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类。"

    partial_fb_provider = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(
            content=json.dumps(
                {
                    "action": "clarify",
                    "answers": {"A1-auth-anon": "公开浏览+登录发帖"},
                    "resolutions": {},
                    "feedback": "所有数据服务端持久化",
                    "message": "Parsed A1 and one note.",
                }
            )
        )
    )
    partial_intent = parse_reply(
        provider=partial_fb_provider,
        message=(
            "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类\n"
            "所有数据服务端持久化\n"
            "A1-auth-anon: 公开浏览+登录发帖"
        ),
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert partial_intent.action == "clarify"
    assert "AI 新闻 / AI 工具 / AI 产品" in partial_intent.feedback
    assert "所有数据服务端持久化" in partial_intent.feedback
    assert partial_intent.feedback.count("所有数据服务端持久化") == 1


def test_resynthesize_concurrent_revision_raises_conflict_without_overwriting(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(title="blacknote", workflow_name="full_sdlc_tdd")
    draft = _draft_spec()
    store.save_spec(run_id, draft, actor="align")
    clarified = RunControl(store).clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "强制登录后访问",
            "A2-media-source": "本地上传或图片 URL",
        },
        actor="human",
    )

    newer_spec = clarified.model_copy(
        update={"version": clarified.version + 1, "goal": "并发更新后的最新目标"}
    )

    def _racy_responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages, tools
        # Simulate another writer persisting a newer spec while LLM is running.
        store.save_spec(run_id, newer_spec, actor="human")
        fused = clarified.model_copy(
            update={
                "constraints": [
                    *clarified.constraints,
                    "强制登录后访问",
                    "话题固定包含 AI 新闻 / AI 工具 / AI 产品",
                ]
            }
        )
        return LLMResponse(content=fused.model_dump_json())

    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=_racy_responder),
        workspace_root=tmp_path,
    )
    with pytest.raises(ConcurrencyConflictError):
        stage.resynthesize_after_clarification(
            run_id=run_id,
            extra_feedback="话题固定包含 AI 新闻 / AI 工具 / AI 产品",
            strict_canonical=True,
        )

    latest = store.get_run(run_id)["spec"]
    assert latest is not None
    assert latest.goal == "并发更新后的最新目标"
    assert not any(ev.event_type == "spec.resynthesized" for ev in store.list_events(run_id))


def test_resynthesize_rejects_unchanged_body_and_misattributed_human_assumptions(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(title="blacknote", workflow_name="full_sdlc_tdd")
    draft = _draft_spec()
    store.save_spec(run_id, draft, actor="align")
    clarified = RunControl(store).clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "强制登录后访问，内置邮箱+密码账号体系",
            "A2-media-source": "本地上传或图片 URL",
        },
        actor="human",
    )

    # First attempt returns unchanged body; second attempt misattributes human decision as agent.
    bad_agent_assumption = clarified.model_copy(
        update={
            "constraints": [*clarified.constraints, "话题固定包含 AI 新闻 / AI 工具 / AI 产品"],
            "assumptions": ["强制登录后访问，内置邮箱+密码账号体系"],
            "agent_high_impact_assumptions": ["强制登录后访问，内置邮箱+密码账号体系"],
        }
    )
    replies = [clarified.model_dump_json(), bad_agent_assumption.model_dump_json()]

    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=lambda _m, _t: LLMResponse(content=replies.pop(0))),
        workspace_root=tmp_path,
    )
    with pytest.raises(ValueError, match="agent_high_impact_assumptions"):
        stage.resynthesize_after_clarification(
            run_id=run_id,
            extra_feedback="话题固定包含 AI 新闻 / AI 工具 / AI 产品",
            strict_canonical=True,
        )

    assert not any(ev.event_type == "spec.resynthesized" for ev in store.list_events(run_id))
    assert any(ev.event_type == "spec.validation_failed" for ev in store.list_events(run_id))


def test_resynthesize_enforces_workflow_strict_canonical_when_draft_lacks_deliverable_kind(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    # Initial draft skipped canonical validation because it had blocking ambiguities.
    incomplete_draft = _draft_spec().model_copy(
        update={"deliverable_kind": None, "observable_surfaces": []}
    )
    still_incomplete = incomplete_draft.model_copy(
        update={
            "constraints": [
                *incomplete_draft.constraints,
                "话题固定为 AI 新闻 / AI 工具 / AI 产品",
            ]
        }
    )

    calls = 0

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal calls
        del tools
        system = messages[0].content
        if "Requirements Alignment Architect" in system:
            calls += 1
            return LLMResponse(
                content=incomplete_draft.model_dump_json()
                if calls == 1
                else still_incomplete.model_dump_json()
            )
        raise AssertionError(f"Should not reach downstream stage: {system[:80]}")

    provider = MockLLMProvider(responder=_responder)
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: provider,
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    replied = runner.reply(
        run_id=started.run_id,
        message=(
            "话题固定为 AI 新闻 / AI 工具 / AI 产品\n"
            "A1-auth-anon: 公开浏览+登录发帖\n"
            "A2-media-source: 本地上传或图片 URL"
        ),
    )
    assert replied.status == SessionStatus.NEEDS_INPUT
    assert replied.pause_kind == PauseKind.BLOCKED
    assert "deliverable_kind" in replied.message
    store = RunStore(state_dir / "olympus.db")
    snap = store.get_run(started.run_id)
    assert snap["design"] is None
    assert not any(
        ev.event_type == "spec.resynthesized" for ev in store.list_events(started.run_id)
    )


def test_session_runner_retries_failed_resynthesis_in_place_preserving_notes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    draft = _draft_spec()
    fused = draft.model_copy(
        update={
            "requirements": [
                RequirementItem(
                    id="R1",
                    actor="AI 工程师",
                    description="平台固定包含 AI 新闻 / AI 工具 / AI 产品 三类一级话题。",
                    ears_criteria=[
                        AcceptanceCriterion(
                            id="AC-1",
                            pattern=EarsPattern.UBIQUITOUS,
                            outcome="展示 AI 新闻 / AI 工具 / AI 产品 三个固定话题筛选入口",
                        )
                    ],
                )
            ]
        }
    )

    propose_calls = 0
    resynth_prompts: list[str] = []

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal propose_calls
        del tools
        system = messages[0].content
        user = messages[1].content
        if "Fuse EVERY human clarification" in system:
            resynth_prompts.append(user)
            if len(resynth_prompts) == 1:
                raise RuntimeError("transient provider outage")
            return LLMResponse(content=fused.model_dump_json())
        if "Requirements Alignment Architect" in system:
            propose_calls += 1
            return LLMResponse(content=draft.model_dump_json())
        if "White-Box Solution Architect" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            assert spec is not None
            return LLMResponse(content=_design_json(spec.spec_id, spec.content_hash))
        if "Task Planner" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            design = snap["design"]
            assert spec is not None and design is not None
            return LLMResponse(
                content=_plan_json(
                    spec.spec_id, spec.content_hash, design.design_id, design.content_hash
                )
            )
        raise AssertionError(f"Unexpected prompt: {system[:80]}")

    provider = MockLLMProvider(responder=_responder)
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: provider,
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    run_id = started.run_id

    failed = runner.reply(
        run_id=run_id,
        message=(
            "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类。\n"
            "A1-auth-anon: 公开浏览+登录发帖\n"
            "A2-media-source: 本地上传或图片 URL"
        ),
    )
    assert failed.status == SessionStatus.NEEDS_INPUT
    assert failed.pause_kind == PauseKind.BLOCKED
    assert "transient provider outage" in failed.message
    assert propose_calls == 1
    assert len(resynth_prompts) == 1

    retried = runner.reply(run_id=run_id, message="retry")
    assert retried.status == SessionStatus.NEEDS_INPUT
    assert retried.pause_kind == PauseKind.PLAN_HANDOFF
    assert propose_calls == 1  # Never re-ran propose_spec from scratch.
    assert len(resynth_prompts) == 2
    assert "话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类。" in resynth_prompts[1]


def test_router_preserves_same_sentence_tls_and_partial_prose_clause() -> None:
    spec = _draft_spec()
    snapshot = {"spec": spec, "plan": None, "status": RunStatus.AWAITING_HUMAN}

    def _never_called(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        raise AssertionError("deterministic fast-path should not call LLM")

    tls_intent = parse_reply(
        provider=MockLLMProvider(responder=_never_called),
        message="Use defaults, and use TLS by default.",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert tls_intent.action == "clarify"
    assert "TLS" in tls_intent.feedback

    prose_provider = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(
            content=json.dumps(
                {
                    "action": "clarify",
                    "answers": {"A1-auth-anon": "强制登录后访问"},
                    "resolutions": {},
                    "feedback": "",
                    "message": "Extracted A1.",
                }
            )
        )
    )
    prose_intent = parse_reply(
        provider=prose_provider,
        message="强制登录后访问，且邮箱验证必须启用",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert prose_intent.action == "clarify"
    assert "邮箱验证必须启用" in prose_intent.feedback


def test_resynthesize_rejects_cosmetic_edit_ignoring_non_default_answer_and_reads_ears(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(title="blacknote", workflow_name="full_sdlc_tdd")
    draft = _draft_spec()
    store.save_spec(run_id, draft, actor="align")
    clarified = RunControl(store).clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "强制登录后访问",
            "A2-media-source": "本地上传或图片 URL",
        },
        clarification_note="邮箱验证必须启用",
        actor="human",
    )

    # Attempt 1: cosmetic period on goal only (still ignores "强制登录后访问" & "邮箱验证").
    cosmetic = clarified.model_copy(update={"goal": clarified.goal + "。"})
    # Attempt 2: faithfully updates EARS outcome with BOTH decisions while keeping
    # acceptance_criteria populated so _materialize_ears_text does not auto-copy ears_criteria.
    ears_fused = clarified.model_copy(
        update={
            "requirements": [
                RequirementItem(
                    id="R1",
                    actor="AI 工程师",
                    description="平台支持强制登录后访问并启用邮箱验证，按话题筛选笔记。",
                    acceptance_criteria=["原有验收标准"],
                    ears_criteria=[
                        AcceptanceCriterion(
                            id="AC-1",
                            pattern=EarsPattern.UBIQUITOUS,
                            outcome="仅允许完成邮箱验证并登录的用户访问信息流",
                        )
                    ],
                )
            ]
        }
    )
    replies = [cosmetic.model_dump_json(), ears_fused.model_dump_json()]
    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=lambda _m, _t: LLMResponse(content=replies.pop(0))),
        workspace_root=tmp_path,
    )
    updated = stage.resynthesize_after_clarification(
        run_id=run_id,
        extra_feedback="邮箱验证必须启用",
        strict_canonical=True,
    )
    assert len(replies) == 0  # First cosmetic attempt was rejected; second EARS-fused succeeded.
    assert updated.version == clarified.version + 1


def test_resynthesize_detects_concurrent_clarification_note(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(title="blacknote", workflow_name="full_sdlc_tdd")
    draft = _draft_spec()
    store.save_spec(run_id, draft, actor="align")
    clarified = RunControl(store).clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "公开浏览+登录发帖",
            "A2-media-source": "本地上传或图片 URL",
        },
        clarification_note="话题固定包含 AI 新闻 / AI 工具 / AI 产品",
        actor="human",
    )

    def _racy_responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages, tools
        store.append_event(
            run_id=run_id,
            stage="align",
            event_type="spec.clarification_note",
            actor="human",
            payload={"spec_id": clarified.spec_id, "feedback": "必须开启邮箱验证"},
        )
        fused = clarified.model_copy(
            update={
                "constraints": [
                    *clarified.constraints,
                    "强制登录后访问",
                    "话题固定包含 AI 新闻 / AI 工具 / AI 产品",
                ]
            }
        )
        return LLMResponse(content=fused.model_dump_json())

    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=_racy_responder),
        workspace_root=tmp_path,
    )
    with pytest.raises(ConcurrencyConflictError):
        stage.resynthesize_after_clarification(
            run_id=run_id,
            extra_feedback="话题固定包含 AI 新闻 / AI 工具 / AI 产品",
            strict_canonical=True,
        )
    assert not any(ev.event_type == "spec.resynthesized" for ev in store.list_events(run_id))


def test_approve_cannot_bypass_pending_resynthesis_after_crash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    draft = _draft_spec()
    fused = draft.model_copy(
        update={
            "requirements": [
                RequirementItem(
                    id="R1",
                    actor="AI 工程师",
                    description="平台固定包含 AI 新闻 / AI 工具 / AI 产品 三类话题。",
                    ears_criteria=[
                        AcceptanceCriterion(
                            id="AC-1",
                            pattern=EarsPattern.UBIQUITOUS,
                            outcome="展示 AI 新闻 / AI 工具 / AI 产品 三个固定话题入口",
                        )
                    ],
                )
            ]
        }
    )
    resynth_called = 0
    design_prompts: list[str] = []

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal resynth_called
        del tools
        system = messages[0].content
        user = messages[1].content
        if "Session Reply Router" in system:
            return LLMResponse(content=json.dumps({"action": "approve", "message": "approve"}))
        if "Fuse EVERY human clarification" in system:
            resynth_called += 1
            return LLMResponse(content=fused.model_dump_json())
        if "Requirements Alignment Architect" in system:
            return LLMResponse(content=draft.model_dump_json())
        if "White-Box Solution Architect" in system:
            design_prompts.append(user)
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            assert spec is not None
            return LLMResponse(content=_design_json(spec.spec_id, spec.content_hash))
        if "Task Planner" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            design = snap["design"]
            assert spec is not None and design is not None
            return LLMResponse(
                content=_plan_json(
                    spec.spec_id, spec.content_hash, design.design_id, design.content_hash
                )
            )
        raise AssertionError(f"Unexpected prompt: {system[:80]}")

    provider = MockLLMProvider(responder=_responder)
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: provider,
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    run_id = started.run_id
    # Simulate process crash immediately after atomic clarify_spec commit, before _continue runs.
    runner.control.clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "公开浏览+登录发帖",
            "A2-media-source": "本地上传或图片 URL",
        },
        clarification_note="话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类",
        actor="human",
    )

    approved = runner.reply(run_id=run_id, message="approve")
    assert approved.status == SessionStatus.NEEDS_INPUT
    assert approved.pause_kind == PauseKind.PLAN_HANDOFF
    assert resynth_called == 1
    assert len(design_prompts) == 1
    assert "AI 新闻 / AI 工具 / AI 产品" in design_prompts[0]


def test_router_does_not_subtract_unanswered_option_words_from_prose() -> None:
    spec = _draft_spec(
        ambiguities=[
            AmbiguityItem(
                id="AMB-CACHE",
                question="Which cache backend should be used?",
                impact="Cache layer",
                options=["Redis", "Memcached"],
                default_assumption="Redis",
                blocking=True,
            ),
            AmbiguityItem(
                id="AMB-TLS",
                question="Should TLS be enabled on internal hops?",
                impact="Transport security",
                options=["TLS on", "TLS off"],
                default_assumption="TLS on",
                blocking=True,
            ),
        ]
    )
    snapshot = {"spec": spec, "plan": None, "status": RunStatus.AWAITING_HUMAN}
    provider = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(
            content=json.dumps(
                {
                    "action": "clarify",
                    "answers": {"AMB-CACHE": "Redis"},
                    "resolutions": {},
                    "feedback": "",
                    "message": "Extracted cache.",
                }
            )
        )
    )
    intent = parse_reply(
        provider=provider,
        message="Use Redis, and keep TLS off.",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert intent.action == "clarify"
    assert "TLS off" in intent.feedback


def test_resynthesize_rejects_negated_or_partial_technical_constraints(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(title="infra", workflow_name="full_sdlc_tdd")
    draft = _draft_spec()
    store.save_spec(run_id, draft, actor="align")
    clarified = RunControl(store).clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "公开浏览+登录发帖",
            "A2-media-source": "本地上传或图片 URL",
        },
        clarification_note="启用 TLS 并支持 PostgreSQL、Redis",
        actor="human",
    )

    negated_only = clarified.model_copy(
        update={"constraints": [*clarified.constraints, "TLS is explicitly disabled."]}
    )
    replies = [negated_only.model_dump_json(), negated_only.model_dump_json()]
    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=lambda _m, _t: LLMResponse(content=replies.pop(0))),
        workspace_root=tmp_path,
    )
    with pytest.raises(ValueError, match="PostgreSQL|Redis|negated"):
        stage.resynthesize_after_clarification(
            run_id=run_id,
            extra_feedback="启用 TLS 并支持 PostgreSQL、Redis",
            strict_canonical=True,
        )


def test_note_appended_after_save_spec_remains_unconsumed_by_event_id(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    draft = _draft_spec()
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: MockLLMProvider(
                responder=lambda _m, _t: LLMResponse(content=draft.model_dump_json())
            ),
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    run_id = started.run_id
    clarified = runner.control.clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "公开浏览+登录发帖",
            "A2-media-source": "本地上传或图片 URL",
        },
        clarification_note="话题固定包含 AI 新闻 / AI 工具 / AI 产品",
        actor="human",
    )
    first_note_id = [
        ev.event_id
        for ev in runner.store.list_events(run_id)
        if ev.event_type == "spec.clarification_note"
    ][0]
    # Simulate an MFA note arriving in the narrow window right before spec.resynthesized commits.
    runner.store.append_event(
        run_id=run_id,
        stage="align",
        event_type="spec.clarification_note",
        actor="human",
        payload={"spec_id": clarified.spec_id, "feedback": "强制启用 MFA 多因素认证"},
    )
    runner.store.append_event(
        run_id=run_id,
        stage="align",
        event_type="spec.resynthesized",
        actor="align",
        payload={
            "spec_id": clarified.spec_id,
            "version": clarified.version + 1,
            "spec_hash": clarified.content_hash,
            "previous_spec_hash": clarified.content_hash,
            "had_extra_feedback": True,
            "consumed_note_ids": [first_note_id],
        },
    )
    remaining = runner._clarification_notes(run_id, spec_id=clarified.spec_id)
    assert remaining == ["强制启用 MFA 多因素认证"]
    assert runner._has_pending_resynthesis(run_id, clarified) is True


def test_approve_reloads_latest_snapshot_when_clarify_lands_concurrently(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    draft = _draft_spec()
    fused = draft.model_copy(
        update={
            "requirements": [
                RequirementItem(
                    id="R1",
                    actor="AI 工程师",
                    description="平台固定包含 AI 新闻 / AI 工具 / AI 产品 三类话题。",
                    ears_criteria=[
                        AcceptanceCriterion(
                            id="AC-1",
                            pattern=EarsPattern.UBIQUITOUS,
                            outcome="固定提供 AI 新闻 / AI 工具 / AI 产品 三个话题入口",
                        )
                    ],
                )
            ]
        }
    )
    resynth_called = 0

    def _responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal resynth_called
        del tools
        system = messages[0].content
        if "Session Reply Router" in system:
            # Concurrent clarification commits while router is running!
            runner.control.clarify_spec(
                run_id=run_id,
                answers={
                    "A1-auth-anon": "公开浏览+登录发帖",
                    "A2-media-source": "本地上传或图片 URL",
                },
                clarification_note="话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类",
                actor="human",
            )
            return LLMResponse(content=json.dumps({"action": "approve", "message": "approve"}))
        if "Fuse EVERY human clarification" in system:
            resynth_called += 1
            return LLMResponse(content=fused.model_dump_json())
        if "Requirements Alignment Architect" in system:
            return LLMResponse(content=draft.model_dump_json())
        if "White-Box Solution Architect" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            assert spec is not None
            return LLMResponse(content=_design_json(spec.spec_id, spec.content_hash))
        if "Task Planner" in system:
            snap = RunStore(state_dir / "olympus.db").get_run(run_id)
            spec = snap["spec"]
            design = snap["design"]
            assert spec is not None and design is not None
            return LLMResponse(
                content=_plan_json(
                    spec.spec_id, spec.content_hash, design.design_id, design.content_hash
                )
            )
        raise AssertionError(f"Unexpected prompt: {system[:80]}")

    provider = MockLLMProvider(responder=_responder)
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: provider,
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    run_id = started.run_id
    out = runner.reply(run_id=run_id, message="approve")
    assert out.status == SessionStatus.NEEDS_INPUT
    assert out.pause_kind == PauseKind.PLAN_HANDOFF
    assert resynth_called == 1
    assert any(
        ev.event_type == "spec.resynthesized" for ev in runner.store.list_events(run_id)
    )


def test_router_preserves_second_clause_even_when_words_overlap_question() -> None:
    spec = _draft_spec(
        ambiguities=[
            AmbiguityItem(
                id="A1",
                question="Which cache backend should handle public access traffic?",
                impact="Cache layer",
                options=["Redis", "Memcached"],
                default_assumption="Redis",
                blocking=True,
            )
        ]
    )
    snapshot = {"spec": spec, "plan": None, "status": RunStatus.AWAITING_HUMAN}
    provider = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(
            content=json.dumps(
                {
                    "action": "clarify",
                    "answers": {"A1": "Redis"},
                    "resolutions": {},
                    "feedback": "",
                    "message": "Extracted A1.",
                }
            )
        )
    )
    intent = parse_reply(
        provider=provider,
        message="Use Redis, and allow public access.",
        snapshot=snapshot,
        pause_kind=PauseKind.CLARIFY,
        gate="spec",
        completed=False,
    )
    assert intent.action == "clarify"
    assert "public access" in intent.feedback


def test_resynthesize_rejects_inverted_negative_constraint(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(title="tls-off", workflow_name="full_sdlc_tdd")
    draft = _draft_spec()
    store.save_spec(run_id, draft, actor="align")
    clarified = RunControl(store).clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "公开浏览+登录发帖",
            "A2-media-source": "本地上传或图片 URL",
        },
        clarification_note="Keep TLS off.",
        actor="human",
    )

    inverted = clarified.model_copy(
        update={"constraints": [*clarified.constraints, "TLS is enabled."]}
    )
    replies = [inverted.model_dump_json(), inverted.model_dump_json()]
    stage = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=lambda _m, _t: LLMResponse(content=replies.pop(0))),
        workspace_root=tmp_path,
    )
    with pytest.raises(ValueError, match="prohibition/disabled"):
        stage.resynthesize_after_clarification(
            run_id=run_id,
            extra_feedback="Keep TLS off.",
            strict_canonical=True,
        )


def test_recover_blocked_note_bumps_revision_blocking_concurrent_approve(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state_dir = tmp_path / "state"
    draft = _draft_spec()
    runner = SessionRunner(
        SessionConfig(
            workspace_root=repo,
            state_dir=state_dir,
            step_mode=StepMode.PLAN_ONLY,
            gate_policy=GatePolicyName.INTERACTIVE_MINIMAL,
            provider="mock",
            provider_builder=lambda **_kwargs: MockLLMProvider(
                responder=lambda _m, _t: LLMResponse(content=draft.model_dump_json())
            ),
        )
    )
    started = runner.start(goal="构建 AI 工程师图文社区")
    run_id = started.run_id
    runner.control.clarify_spec(
        run_id=run_id,
        answers={
            "A1-auth-anon": "公开浏览+登录发帖",
            "A2-media-source": "本地上传或图片 URL",
        },
        actor="human",
    )
    rev_before = int(runner.store.get_run(run_id)["state_revision"])
    runner.control.clarify_spec(
        run_id=run_id,
        answers={},
        clarification_note="Require MFA.",
        actor="human",
    )
    rev_after = int(runner.store.get_run(run_id)["state_revision"])
    assert rev_after > rev_before
    with pytest.raises(ConcurrencyConflictError):
        runner._approve_gate(run_id, "spec", expected_revision=rev_before)
