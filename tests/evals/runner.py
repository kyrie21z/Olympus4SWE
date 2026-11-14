"""End-to-end Stage 1 eval runner with isolated repos, oracle loops, and artifacts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from olympus.core.contracts import CritiqueSeverity, GoalSpec, RunMode, TaskPlan
from olympus.core.dag import PlanValidationError
from olympus.core.store import RunStore
from olympus.providers.factory import build_provider
from olympus.providers.types import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    MockLLMProvider,
    ToolSpec,
)
from olympus.stages.align import AlignStage
from olympus.stages.orchestrate import OrchestrateStage
from tests.evals.oracle import (
    LLMOracle,
    OracleProtocol,
    QuestionTracker,
    RuleBasedOracle,
    normalize_question,
)
from tests.evals.report import generate_reports, read_records
from tests.evals.schemas import (
    BenchmarkClass,
    CaseRunRecord,
    EvalCase,
    JudgeResult,
    RepoFileFixture,
    RubricMetrics,
    RunMeta,
    SuiteReport,
    load_eval_cases,
)
from tests.evals.scorers import (
    CaseEvidence,
    LLMJudge,
    evaluate_case_outcome,
)

EvalMode = Literal["mock", "live"]
_SUPPORTED_VARIANTS = frozenset({"baseline", "critique"})
_SPEC_BINDING_RE = re.compile(r"Set spec_id='([^']+)' and spec_hash='([^']+)'")


class CountingProvider:
    """Thread-safe wrapper counting calls and tokens for any provider."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.model_name = inner.model_name
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
    ) -> LLMResponse:
        response = self._inner.complete(
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
        return response


def _mock_ambiguity(
    slot_id: str,
    topic: str,
    keywords: Sequence[str],
    blocking: bool,
) -> dict[str, Any]:
    keyword = keywords[0] if keywords else "具体要求"
    return {
        "id": f"AMB-{slot_id}",
        "question": f"{topic}：请明确 {keyword} 的具体要求？",
        "impact": "影响验收口径与实现边界",
        "options": [],
        "blocking": blocking,
    }


def _slot_ambiguities(case: EvalCase) -> list[dict[str, Any]]:
    return [
        _mock_ambiguity(slot.slot_id, slot.topic, slot.keywords, slot.blocking)
        for slot in case.hidden_slots
    ]


def _first_python_path(case: EvalCase, *, prefix: str) -> str:
    for fixture in case.repo_files:
        if fixture.path.endswith(".py") and fixture.path.startswith(prefix):
            return fixture.path
    return case.repo_files[0].path if case.repo_files else "src/app.py"


def _top_dir(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "."


def _mock_spec_payload(case: EvalCase) -> dict[str, Any]:
    if case.benchmark_class == BenchmarkClass.A:
        return {
            "goal": case.raw_goal,
            "requirements": [
                {
                    "id": "R1",
                    "description": "在信息不足或校验失败时暂停并请求澄清，不臆造仓库文件",
                    "acceptance_criteria": [
                        "无法形成可机读验收条件时保持 draft",
                        "open_ambiguities 至少包含 1 个 blocking=true 的条目",
                    ],
                },
                {
                    "id": "R2",
                    "description": "检测到目标与仓库现状冲突时报告错误并等待人类裁决",
                    "acceptance_criteria": ["阻塞项未清零前不得进入编排阶段"],
                },
            ],
            "out_of_scope": ["未经澄清的任何实现工作", "引入新的第三方依赖"],
            "constraints": ["信息不足时必须先暂停"],
            "assumptions": [],
            "open_ambiguities": _slot_ambiguities(case)
            or [_mock_ambiguity("BASE-1", "目标范围与成功标准", ["范围"], True)],
            "minority_notes": [],
        }

    if case.benchmark_class == BenchmarkClass.B:
        source = _first_python_path(case, prefix="src/")
        requirements = [
            {
                "id": "R1",
                "description": f"在 `{source}` 中实现目标能力：{case.raw_goal}",
                "acceptance_criteria": [
                    f"`python3 -m compileall -q {_top_dir(source)}` 退出码为 0",
                    "存储不可用时拒绝请求并返回 503",
                ],
            },
            {
                "id": "R2",
                "description": "覆盖失败路径：超时、重复请求与无效参数",
                "acceptance_criteria": ["非法输入返回 400 且记录错误日志"],
            },
        ]
        constraints = list(case.initial_constraints)
        if case.expected.require_repo_conflict_check and case.expected.expected_repo_conflict_terms:
            constraints.append(
                "仓库现状：" + "、".join(case.expected.expected_repo_conflict_terms)
            )
        return {
            "goal": case.raw_goal,
            "requirements": requirements,
            "out_of_scope": ["重构无关模块", "新增第三方依赖"],
            "constraints": constraints,
            "assumptions": ["仓库现有测试可作为回归基线"],
            "open_ambiguities": _slot_ambiguities(case),
            "minority_notes": [],
        }

    source = _first_python_path(case, prefix="src/")
    test_file = _first_python_path(case, prefix="tests/")
    if case.benchmark_class == BenchmarkClass.C:
        requirements = [
            {
                "id": "R1",
                "description": f"在 `{source}` 中完成：{case.raw_goal}",
                "acceptance_criteria": [
                    f"`python3 -m compileall -q {_top_dir(source)}` 退出码为 0",
                    "输入非法时抛出 ValueError",
                ],
            },
            {
                "id": "R2",
                "description": f"更新 `{test_file}` 覆盖该行为；测试失败时必须报告错误",
                "acceptance_criteria": [f"`{test_file}` 中存在针对该行为的断言"],
            },
        ]
    else:
        requirements = [
            {
                "id": "R1",
                "description": f"实现源码改动：{case.raw_goal}",
                "acceptance_criteria": [
                    "`python3 -m compileall -q src` 退出码为 0",
                    "非法参数返回明确错误码",
                ],
            },
            {
                "id": "R2",
                "description": "补充单元测试覆盖失败路径与边界条件",
                "acceptance_criteria": [
                    "测试命令退出码为 0",
                    "失败场景必须断言异常或错误码",
                ],
            },
            {
                "id": "R3",
                "description": "更新相关文档说明新行为",
                "acceptance_criteria": [
                    "文档包含可执行的示例命令",
                    "文档不得声称未实现的能力",
                ],
            },
        ]
    return {
        "goal": case.raw_goal,
        "requirements": requirements,
        "out_of_scope": ["重构无关模块", "新增第三方依赖", "修改公共 API 签名"],
        "constraints": list(case.initial_constraints),
        "assumptions": [],
        "open_ambiguities": _slot_ambiguities(case),
        "minority_notes": [],
    }


def _task_dirs(case: EvalCase) -> list[str]:
    preferred = ["src", "tests", "docs"]
    dirs: list[str] = []
    for fixture in case.repo_files:
        if "/" not in fixture.path:
            continue
        top = fixture.path.split("/", 1)[0]
        if top not in dirs:
            dirs.append(top)
    return sorted(
        dirs,
        key=lambda name: (preferred.index(name) if name in preferred else len(preferred), name),
    )


def _acceptance_check(directory: str, case: EvalCase) -> dict[str, Any]:
    has_python = any(
        fixture.path.startswith(f"{directory}/") and fixture.path.endswith(".py")
        for fixture in case.repo_files
    )
    if has_python:
        command = f"python3 -m compileall -q {directory}"
    else:
        command = (
            "python3 -c \"import pathlib; "
            f"assert pathlib.Path('{directory}').is_dir()\""
        )
    return {
        "id": f"CHK-{directory}",
        "kind": "command",
        "description": f"Validate {directory}",
        "command": command,
    }


def _mock_plan_payload(case: EvalCase, messages: Sequence[ChatMessage]) -> dict[str, Any]:
    user_prompt = messages[1].content if len(messages) > 1 else ""
    binding = _SPEC_BINDING_RE.search(user_prompt)
    spec_id = binding.group(1) if binding else ""
    spec_hash = binding.group(2) if binding else ""

    requirements = _mock_spec_payload(case)["requirements"]
    requirement_ids = [str(item["id"]) for item in requirements]
    directories = _task_dirs(case)
    tasks: list[dict[str, Any]] = []
    if len(directories) <= 1:
        scope = [f"{directories[0]}/"] if directories else ["."]
        tasks.append(
            {
                "task_id": "T1",
                "title": "Implement requested change",
                "objective": f"Implement: {case.raw_goal}",
                "requirement_refs": requirement_ids,
                "depends_on": [],
                "write_scope": scope,
                "read_scope": [],
                "acceptance_checks": [_acceptance_check(directories[0], case)]
                if directories
                else [
                    {
                        "id": "CHK-ROOT",
                        "kind": "command",
                        "description": "Validate repository root",
                        "command": "python3 -m compileall -q .",
                    }
                ],
                "max_cycles": 2,
            }
        )
    else:
        for index, directory in enumerate(directories):
            refs = [
                requirement_ids[position]
                for position in range(index, len(requirement_ids), len(directories))
            ]
            if not refs:
                refs = [requirement_ids[-1]]
            tasks.append(
                {
                    "task_id": f"T{index + 1}",
                    "title": f"Update {directory}",
                    "objective": f"Implement the {directory} portion of: {case.raw_goal}",
                    "requirement_refs": refs,
                    "depends_on": [],
                    "write_scope": [f"{directory}/"],
                    "read_scope": [],
                    "acceptance_checks": [_acceptance_check(directory, case)],
                    "max_cycles": 2,
                }
            )
    return {
        "plan_id": f"plan-{case.case_id}",
        "spec_id": spec_id,
        "spec_hash": spec_hash,
        "version": 1,
        "tasks": tasks,
    }


def _mock_critique_payload(system_prompt: str) -> dict[str, Any]:
    perspective = "product"
    if "engineering" in system_prompt:
        perspective = "engineering"
    elif "qa" in system_prompt:
        perspective = "qa"
    return {
        "notes": [
            {
                "id": f"CRIT-{perspective.upper()}-1",
                "perspective": perspective,
                "severity": "warning",
                "summary": "Consider documenting operational constraints explicitly.",
            }
        ]
    }


def build_mock_eval_provider(case: EvalCase, variant: str) -> MockLLMProvider:
    """Scripted provider that exercises Align (incl. critique) and Orchestrate offline."""

    def responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        system = messages[0].content if messages else ""
        if "independent evaluator" in system:
            return LLMResponse(
                content=json.dumps(
                    {"met": True, "confidence": 0.9, "reason": "mock judge verdict"},
                    ensure_ascii=False,
                )
            )
        if "Requirements Alignment Architect" in system:
            return LLMResponse(
                content=json.dumps(_mock_spec_payload(case), ensure_ascii=False)
            )
        if "Task Planner" in system:
            return LLMResponse(
                content=json.dumps(_mock_plan_payload(case, messages), ensure_ascii=False)
            )
        if "cold-start read-only" in system:
            return LLMResponse(
                content=json.dumps(_mock_critique_payload(system), ensure_ascii=False)
            )
        raise AssertionError(f"unexpected mock prompt: {system[:120]}")

    return MockLLMProvider(responder=responder, model_name=f"mock-eval-{variant}")


@dataclass(frozen=True)
class ClarifyOutcome:
    spec: GoalSpec
    questions: tuple[str, ...]
    rounds: int
    repeated_rate: float


def _install_git_repo(repo: Path, fixtures: Sequence[RepoFileFixture]) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    root = repo.resolve()
    for fixture in fixtures:
        target = (repo / fixture.path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"repo fixture escapes workspace: {fixture.path}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fixture.content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "eval@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Eval Harness"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "eval fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _empty_metrics(note: str) -> RubricMetrics:
    return RubricMetrics(
        bounded_scope=0.0,
        unresolved_blockers_ok=0.0,
        unresolved_blockers_count=0,
        failure_modes=0.0,
        falsifiable_criteria=0.0,
        downstream_consistency=0.0,
        vague_goal_refusal=0.0,
        ambiguity_recall=0.0,
        ask_precision=0.0,
        over_asking_penalty=0.0,
        leakage_detected=False,
        bounded_turns_ok=0.0,
        total_score=0.0,
        questions_asked=0,
        clarify_rounds=0,
        repeated_question_rate=0.0,
        notes=[note],
    )


class EvalRunner:
    """Runs the four-class benchmark over isolated temporary git repositories."""

    def __init__(
        self,
        *,
        dataset_path: Path,
        output_dir: Path,
        mode: EvalMode = "mock",
        variants: Sequence[str] = ("baseline", "critique"),
        classes: Sequence[BenchmarkClass] | None = None,
        case_ids: Sequence[str] | None = None,
        trials: int = 1,
        with_judge: bool = False,
        provider: str = "auto",
        model: str | None = None,
        request_timeout: float = 240.0,
        input_price_per_1m: float = 0.0,
        output_price_per_1m: float = 0.0,
        resume_dir: Path | None = None,
    ) -> None:
        if trials < 1:
            raise ValueError("trials must be >= 1")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        if input_price_per_1m < 0 or output_price_per_1m < 0:
            raise ValueError("token prices must be non-negative")
        if mode not in ("mock", "live"):
            raise ValueError(f"unknown eval mode: {mode}")

        deduped_variants = list(dict.fromkeys(variants))
        if not deduped_variants:
            raise ValueError("at least one variant is required")
        unknown_variants = sorted(set(deduped_variants) - _SUPPORTED_VARIANTS)
        if unknown_variants:
            raise ValueError(f"unsupported variants: {', '.join(unknown_variants)}")

        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.mode: EvalMode = mode
        self.variants = deduped_variants
        self.classes = list(classes) if classes is not None else None
        self.case_ids = list(case_ids) if case_ids is not None else None
        self.trials = trials
        self.with_judge = with_judge
        self.provider_name = provider
        self.model = model
        self.request_timeout = request_timeout
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m
        self.resume_dir = resume_dir
        self.cases = self._select_cases()
        self._live_provider: LLMProvider | None = None

    def _select_cases(self) -> list[EvalCase]:
        all_cases = load_eval_cases(self.dataset_path)
        selected = all_cases
        if self.classes is not None:
            allowed = set(self.classes)
            selected = [case for case in selected if case.benchmark_class in allowed]
        if self.case_ids is not None:
            requested = set(self.case_ids)
            known = {case.case_id for case in all_cases}
            missing = sorted(requested - known)
            if missing:
                raise ValueError(f"unknown case ids: {', '.join(missing)}")
            selected = [case for case in selected if case.case_id in requested]
        if not selected:
            raise ValueError("no eval cases selected")
        return selected

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self) -> SuiteReport:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "live":
            self._live_provider = build_provider(
                provider=self.provider_name,
                model=self.model,
                timeout=self.request_timeout,
            )

        records_path = self.output_dir / "records.jsonl"
        prior = self._prepare_records(records_path)
        completed = {(record.case_id, record.variant, record.trial) for record in prior}

        new_records: list[CaseRunRecord] = []
        for case in self.cases:
            for variant in self.variants:
                for trial in range(1, self.trials + 1):
                    if (case.case_id, variant, trial) in completed:
                        continue
                    record = self._run_case(case, variant, trial)
                    self._append_record(records_path, record)
                    new_records.append(record)

        selected_ids = {case.case_id for case in self.cases}
        all_records = [
            record
            for record in prior
            if record.case_id in selected_ids and record.variant in self.variants
        ] + new_records
        meta = self._run_meta()
        return generate_reports(output_dir=self.output_dir, meta=meta, records=all_records)

    def dry_run(self) -> dict[str, Any]:
        """Validate fixtures, materialize/clean a temp repo, and estimate call bounds."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        materialized_files = 0
        for case in self.cases:
            with tempfile.TemporaryDirectory(prefix="olympus-eval-dry-") as tmp:
                repo = Path(tmp) / "repo"
                _install_git_repo(repo, case.repo_files)
                for fixture in case.repo_files:
                    if not (repo / fixture.path).is_file():
                        raise RuntimeError(f"failed to materialize {fixture.path}")
                materialized_files += len(case.repo_files)

        summary: dict[str, Any] = {
            "mode": self.mode,
            "dataset": self.dataset_path.name,
            "cases": len(self.cases),
            "case_ids": [case.case_id for case in self.cases],
            "classes": sorted({case.benchmark_class.value for case in self.cases}),
            "variants": self.variants,
            "trials": self.trials,
            "repo_files": materialized_files,
            "estimated_max_llm_calls": self._estimate_max_calls(),
        }
        (self.output_dir / "dry_run.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _prepare_records(self, records_path: Path) -> list[CaseRunRecord]:
        if self.resume_dir is None:
            records_path.write_text("", encoding="utf-8")
            return []
        prior_path = self.resume_dir / "records.jsonl"
        prior = read_records(prior_path)
        if prior_path.resolve() == records_path.resolve():
            return prior
        records_path.write_text("", encoding="utf-8")
        if prior_path.exists():
            shutil.copy2(prior_path, records_path)
        return prior

    def _append_record(self, path: Path, record: CaseRunRecord) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def _run_meta(self) -> RunMeta:
        return RunMeta(
            mode=self.mode,
            dataset=self.dataset_path.name,
            variants=self.variants,
            case_ids=[case.case_id for case in self.cases],
            trials=self.trials,
            provider="mock" if self.mode == "mock" else self.provider_name,
            model=self.model or ("mock-eval" if self.mode == "mock" else None),
            judge=self.with_judge,
            generated_at=datetime.now(UTC).isoformat(),
            input_price_per_1m=self.input_price_per_1m,
            output_price_per_1m=self.output_price_per_1m,
        )

    def _estimate_max_calls(self) -> int:
        per_trial = 0
        for variant in self.variants:
            for case in self.cases:
                per_trial += 2  # proposer + one structured-parse retry
                if variant == "critique":
                    per_trial += 6  # three perspectives x one retry each
                if case.expected.require_downstream_plan_valid:
                    per_trial += 2  # planner + one validation retry
                if self.mode == "live" and not case.expected.expect_refusal_or_pause:
                    per_trial += min(
                        case.expected.max_clarify_rounds,
                        case.expected.max_questions,
                    )
                if self.with_judge:
                    per_trial += 1
        return per_trial * self.trials

    def _estimate_cost(self, counting: CountingProvider) -> float:
        return (
            counting.input_tokens / 1_000_000 * self.input_price_per_1m
            + counting.output_tokens / 1_000_000 * self.output_price_per_1m
        )

    def _run_case(self, case: EvalCase, variant: str, trial: int) -> CaseRunRecord:
        started = time.perf_counter()
        counting: CountingProvider | None = None
        try:
            base_provider = (
                build_mock_eval_provider(case, variant)
                if self.mode == "mock"
                else self._require_live_provider()
            )
            counting = CountingProvider(base_provider)
            return self._run_case_in_dir(case, variant, trial, started, counting)
        except Exception as exc:
            duration = time.perf_counter() - started
            return CaseRunRecord(
                case_id=case.case_id,
                benchmark_class=case.benchmark_class,
                variant=variant,
                trial=trial,
                passed=False,
                total_score=0.0,
                metrics=_empty_metrics(f"runner error: {exc}"),
                judge=JudgeResult(status="skipped", reason="run failed before judging"),
                llm_calls=counting.llm_calls if counting is not None else 0,
                input_tokens=counting.input_tokens if counting is not None else 0,
                output_tokens=counting.output_tokens if counting is not None else 0,
                estimated_cost_usd=self._estimate_cost(counting) if counting is not None else 0.0,
                duration_seconds=duration,
                error=str(exc),
            )

    def _run_case_in_dir(
        self,
        case: EvalCase,
        variant: str,
        trial: int,
        started: float,
        counting: CountingProvider,
    ) -> CaseRunRecord:
        with tempfile.TemporaryDirectory(prefix="olympus-eval-") as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "repo"
            _install_git_repo(repo, case.repo_files)
            store = RunStore(tmp_path / "state" / "eval.db")
            run_id = store.create_run(
                title=f"{case.case_id}:{variant}:{trial}",
                mode=RunMode.SIMULATION,
            )

            oracle = self._build_oracle(case, counting)
            align = AlignStage(store=store, provider=counting, workspace_root=repo)

            initial_spec = align.propose_spec(
                run_id=run_id,
                raw_goal=case.raw_goal,
                constraints=list(case.initial_constraints),
                enable_critique=variant == "critique",
            )
            initial_questions = tuple(
                ambiguity.question for ambiguity in initial_spec.open_ambiguities
            )

            is_refusal = case.expected.expect_refusal_or_pause
            refusal_confirm_rejected = False
            final_confirmable = False
            final_spec = initial_spec
            questions: tuple[str, ...] = _unique_questions(initial_questions)
            rounds = 0
            repeated_rate = 0.0

            if is_refusal:
                try:
                    confirmed = align.confirm_spec(run_id=run_id, approver="eval-harness")
                except ValueError:
                    refusal_confirm_rejected = True
                else:
                    final_spec = confirmed
                    final_confirmable = True
            else:
                outcome = self._clarify(
                    align=align,
                    store=store,
                    run_id=run_id,
                    case=case,
                    oracle=oracle,
                )
                questions = _unique_questions((*initial_questions, *outcome.questions))
                rounds = outcome.rounds
                repeated_rate = outcome.repeated_rate
                try:
                    final_spec = align.confirm_spec(run_id=run_id, approver="eval-harness")
                    final_confirmable = True
                except ValueError:
                    final_spec = _current_spec(store, run_id)

            plan: TaskPlan | None = None
            plan_error: str | None = None
            if case.expected.require_downstream_plan_valid and final_confirmable:
                orchestrate = OrchestrateStage(store=store, provider=counting, workspace_root=repo)
                try:
                    plan = orchestrate.propose_plan(run_id=run_id)
                except PlanValidationError as exc:
                    plan_error = "; ".join(exc.violations)
                except Exception as exc:
                    plan_error = str(exc)

            judge: JudgeResult
            if self.with_judge and final_confirmable and not is_refusal:
                judge = LLMJudge(provider=counting).judge(spec=final_spec, case=case)
            else:
                judge = JudgeResult(
                    status="skipped",
                    reason="judge disabled or not applicable to this case",
                )

            evidence = CaseEvidence(
                case=case,
                initial_spec=initial_spec,
                final_spec=final_spec,
                refusal_confirm_rejected=refusal_confirm_rejected,
                final_confirmable=final_confirmable,
                asked_questions=questions,
                clarify_rounds=rounds,
                repeated_question_rate=repeated_rate,
                revealed_slot_ids=frozenset(oracle.revealed_slot_ids),
                oracle_leakage_detected=oracle.leakage_detected,
                oracle_leak_terms=tuple(oracle.leak_terms),
                plan=plan,
                workspace_root=repo,
            )
            passed, total, metrics = evaluate_case_outcome(evidence)
            cost = self._estimate_cost(counting)
            return CaseRunRecord(
                case_id=case.case_id,
                benchmark_class=case.benchmark_class,
                variant=variant,
                trial=trial,
                passed=passed,
                total_score=total,
                metrics=metrics,
                judge=judge,
                llm_calls=counting.llm_calls,
                input_tokens=counting.input_tokens,
                output_tokens=counting.output_tokens,
                estimated_cost_usd=cost,
                duration_seconds=time.perf_counter() - started,
                error=plan_error,
            )

    def _clarify(
        self,
        *,
        align: AlignStage,
        store: RunStore,
        run_id: str,
        case: EvalCase,
        oracle: OracleProtocol,
    ) -> ClarifyOutcome:
        tracker = QuestionTracker()
        questions: list[str] = []
        rounds = 0
        spec = _current_spec(store, run_id)
        for _ in range(case.expected.max_clarify_rounds):
            spec = _current_spec(store, run_id)
            pending = [ambiguity for ambiguity in spec.open_ambiguities if not ambiguity.answer]
            if not pending:
                break
            batch = [ambiguity.question for ambiguity in pending]
            tracker.record(batch)
            questions.extend(batch)
            replies = oracle.reply(pending)
            answers = {
                ambiguity.id: reply.answer
                for ambiguity, reply in zip(pending, replies, strict=True)
            }
            resolutions = {
                note.id: "Marked resolved during the eval clarification round."
                for note in spec.minority_notes
                if note.severity == CritiqueSeverity.BLOCKING and not note.resolved
            }
            spec = align.apply_clarifications(
                run_id=run_id,
                answers=answers,
                resolutions=resolutions,
                actor="eval-oracle",
            )
            rounds += 1
            if not spec.unresolved_blockers():
                break
        return ClarifyOutcome(
            spec=spec,
            questions=tuple(questions),
            rounds=rounds,
            repeated_rate=tracker.repeated_rate,
        )

    def _build_oracle(self, case: EvalCase, provider: LLMProvider) -> OracleProtocol:
        if self.mode == "mock":
            return RuleBasedOracle(
                case.hidden_slots,
                max_questions=case.expected.max_questions,
            )
        return LLMOracle(
            provider=provider,
            slots=case.hidden_slots,
            max_questions=case.expected.max_questions,
        )

    def _require_live_provider(self) -> LLMProvider:
        if self._live_provider is None:
            raise RuntimeError("live provider is not initialized")
        return self._live_provider


def _current_spec(store: RunStore, run_id: str) -> GoalSpec:
    snapshot: Mapping[str, Any] = store.get_run(run_id)
    spec = snapshot["spec"]
    if not isinstance(spec, GoalSpec):
        raise ValueError(f"run {run_id} has no GoalSpec")
    return spec


def _unique_questions(questions: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for question in questions:
        key = normalize_question(question)
        if key in seen:
            continue
        seen.add(key)
        unique.append(question)
    return tuple(unique)
