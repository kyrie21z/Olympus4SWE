"""Stage 1 (Align): Evidence-backed spec synthesis, optional critique, and human freeze."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import Field

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CritiqueNote,
    CritiquePerspective,
    GoalSpec,
    RunStatus,
    SpecStatus,
    StrictModel,
)
from olympus.core.store import RunStore
from olympus.harness.loop import AgentLoop
from olympus.providers.base import LLMProvider
from olympus.stages.common import format_workspace_inventory, run_structured_loop
from olympus.tools.aci import WorkspaceToolkit


class CritiqueOutput(StrictModel):
    notes: list[CritiqueNote] = Field(default_factory=list)


_ALIGN_SYSTEM_PROMPT = """You are the Stage 1 Requirements Alignment Architect of Olympus4SWE.
Inspect the observed workspace context (and use read-only tools when deeper inspection is helpful)
to convert the user's raw goal into a crisp, verifiable GoalSpec JSON object.

Rules:
1. Strict Path Grounding: Cite ONLY file paths listed in Observed workspace files or observed
   via tool output. NEVER invent or mention unverified paths (e.g., `src/.../main.py` or
   `tests/...`) inside `constraints`, `assumptions`, `open_ambiguities`, or `minority_notes`.
   When a task is standalone or target layout is unspecified, refer to behavior/deliverables
   conceptually rather than guessing repository paths.
2. Semantic Audit & Genuine Blockers: Inspect the goal (and existing code via read-only tools)
   across each distinct behavioral dimension — contradictions with the repo, missing high-level
   scope/success criteria (vague goals MUST stay blocked via `open_ambiguities`), undefined value
   domains, empty/boundary/error/status contracts, operation ordering/tie-breakers/idempotency,
   schema/output format/precision/serialization, optimization targets, and persistence/compatibility
   defaults — so no correctness-critical choice is silently guessed.
3. Question Austerity (NO Shotgun Questioning): Treat clarification budget as precious.
   - If requirements are already clear and verifiable, return `"open_ambiguities": []`.
   - When clarifications are genuinely required, cover every distinct decision blocker
     (`blocking: true`, typically 1-4 focused items): keep independent behavioral dimensions
     separate when merging would leave a sub-decision ambiguous, and merge only truly overlapping
     points.
   - NEVER pad with generic boilerplate questions about file paths/layout, language version,
     CLI vs function signature, logging, test framework, or validation of inputs already ruled
     out by stated constraints.
4. Bounded Scope & Falsifiable Criteria: List explicit `out_of_scope` items to prevent scope
   creep, and give every requirement concrete, verifiable `acceptance_criteria`.
5. Exploration Discipline: Limit read-only repo checks to 2-4 focused calls (public API/module
   layout only) before returning GoalSpec JSON; leave deep implementation reading to Stage 2.
6. Output ONLY a single valid JSON object conforming to GoalSpec. Never emit `$comment`,
   `$schema`, or unlisted extra keys."""

_CRITIQUE_SYSTEM_TEMPLATE = """You are a cold-start read-only {perspective} reviewer.
Inspect the draft GoalSpec (and repository files if needed) and report ONLY real omissions,
hidden contradictions, feasibility risks, or untestable acceptance criteria.
Do NOT invent scores or rewrite the spec. If no issues exist, return {{"notes": []}}.
Output ONLY valid JSON matching {{"notes": [...]}} with no `$comment`, `$schema`, or extra keys."""


class AlignStage:
    """First-principles requirements alignment: probe -> optional critique -> clarify -> freeze."""

    def __init__(
        self,
        *,
        store: RunStore,
        provider: LLMProvider,
        workspace_root: Path | str,
        trace_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.workspace_root = Path(workspace_root).resolve()
        self.trace_dir = trace_dir

    def _trace_file(self, run_id: str, name: str) -> Path | None:
        if self.trace_dir is None:
            return None
        return self.trace_dir / run_id / f"{name}.jsonl"

    def propose_spec(
        self,
        *,
        run_id: str,
        raw_goal: str,
        constraints: list[str] | None = None,
        enable_critique: bool = False,
        max_steps: int = 10,
    ) -> GoalSpec:
        """Probe repo context, draft GoalSpec, and optionally run cold-start critique."""
        self.store.set_run_status(run_id, RunStatus.ALIGNING, actor="align", stage="align")
        toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        loop = AgentLoop(
            provider=self.provider,
            toolkit=toolkit,
            store=self.store,
            run_id=run_id,
            actor="align:proposer",
            max_steps=max_steps,
            trace_path=self._trace_file(run_id, "align-proposer"),
        )
        inventory = format_workspace_inventory(toolkit)
        schema_json = json.dumps(GoalSpec.model_json_schema(), ensure_ascii=False)
        user_prompt = (
            f"Workspace root: {self.workspace_root}\n"
            f"{inventory}\n"
            f"Raw user goal:\n{raw_goal}\n\n"
            f"Explicit constraints: {json.dumps(constraints or [], ensure_ascii=False)}\n\n"
            f"Return GoalSpec JSON matching schema:\n{schema_json}"
        )
        draft = run_structured_loop(
            loop=loop,
            system_prompt=_ALIGN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_cls=GoalSpec,
        )

        notes: list[CritiqueNote] = list(draft.minority_notes)
        if enable_critique:
            notes.extend(self._run_parallel_critique(run_id=run_id, spec=draft))

        spec = draft.model_copy(
            update={
                "minority_notes": notes,
                "status": SpecStatus.AWAITING_HUMAN,
            }
        )
        self.store.save_spec(run_id, spec, actor="align")
        self.store.set_run_status(
            run_id,
            RunStatus.AWAITING_HUMAN,
            actor="align",
            stage="align",
            detail={
                "spec_hash": spec.content_hash,
                "blockers": spec.unresolved_blockers(),
            },
        )
        return spec

    def _run_parallel_critique(self, *, run_id: str, spec: GoalSpec) -> list[CritiqueNote]:
        """Run isolated cold-start read-only critique perspectives in parallel."""
        perspectives = (
            CritiquePerspective.PRODUCT,
            CritiquePerspective.ENGINEERING,
            CritiquePerspective.QA,
        )

        critique_schema = json.dumps(CritiqueOutput.model_json_schema(), ensure_ascii=False)

        def run_one(perspective: CritiquePerspective) -> list[CritiqueNote]:
            toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
            loop = AgentLoop(
                provider=self.provider,
                toolkit=toolkit,
                store=self.store,
                run_id=run_id,
                actor=f"align:critic-{perspective.value}",
                max_steps=6,
                trace_path=self._trace_file(run_id, f"align-critic-{perspective.value}"),
            )
            inventory = format_workspace_inventory(toolkit)
            sys_prompt = _CRITIQUE_SYSTEM_TEMPLATE.format(perspective=perspective.value)
            user_prompt = (
                f"Workspace root: {self.workspace_root}\n"
                f"{inventory}\n"
                f"Draft GoalSpec:\n{spec.model_dump_json(indent=2)}\n\n"
                f"Critique ONLY from the {perspective.value} perspective. "
                f"Set perspective='{perspective.value}' on every note.\n"
                f"Return JSON matching schema:\n{critique_schema}"
            )
            out = run_structured_loop(
                loop=loop,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                model_cls=CritiqueOutput,
            )
            return out.notes

        collected: list[CritiqueNote] = []
        with ThreadPoolExecutor(max_workers=len(perspectives)) as pool:
            for notes in pool.map(run_one, perspectives):
                collected.extend(notes)
        return collected

    def apply_clarifications(
        self,
        *,
        run_id: str,
        answers: dict[str, str],
        resolutions: dict[str, str] | None = None,
        extra_out_of_scope: list[str] | None = None,
        actor: str = "zeus",
    ) -> GoalSpec:
        """Apply human answers/resolutions, producing a new spec version (invalidates old hash)."""
        snap = self.store.get_run(run_id)
        spec: GoalSpec | None = snap["spec"]
        if spec is None:
            raise ValueError(f"Run {run_id} has no GoalSpec to clarify")

        updated_ambiguities = [
            amb.model_copy(update={"answer": answers[amb.id]}) if amb.id in answers else amb
            for amb in spec.open_ambiguities
        ]
        res_map = resolutions or {}
        updated_notes = [
            note.model_copy(
                update={"resolved": True, "resolution_note": res_map[note.id]}
            )
            if note.id in res_map
            else note
            for note in spec.minority_notes
        ]
        merged_oos = list(dict.fromkeys([*spec.out_of_scope, *(extra_out_of_scope or [])]))
        next_spec = spec.model_copy(
            update={
                "version": spec.version + 1,
                "open_ambiguities": updated_ambiguities,
                "minority_notes": updated_notes,
                "out_of_scope": merged_oos,
                "status": SpecStatus.AWAITING_HUMAN,
            }
        )
        self.store.save_spec(run_id, next_spec, actor=actor)
        self.store.append_event(
            run_id=run_id,
            stage="align",
            event_type="spec.clarified",
            actor=actor,
            payload={
                "version": next_spec.version,
                "spec_hash": next_spec.content_hash,
                "remaining_blockers": next_spec.unresolved_blockers(),
            },
        )
        return next_spec

    def confirm_spec(
        self,
        *,
        run_id: str,
        approver: str = "zeus",
        accept_risk: bool = False,
        reason: str = "",
        commit_sha: str | None = None,
    ) -> GoalSpec:
        """Freeze and approve the current GoalSpec once all blocking ambiguities are resolved."""
        snap = self.store.get_run(run_id)
        spec: GoalSpec | None = snap["spec"]
        if spec is None:
            raise ValueError(f"Run {run_id} has no GoalSpec to confirm")

        blockers = spec.unresolved_blockers()
        if blockers and not accept_risk:
            raise ValueError(
                f"Cannot confirm spec with unresolved blockers: {', '.join(blockers)}. "
                "Resolve them via clarify or explicitly pass accept_risk=True with a reason."
            )

        decision = ApprovalDecision.ACCEPT_RISK if accept_risk else ApprovalDecision.APPROVED
        confirmed = spec.model_copy(update={"status": SpecStatus.CONFIRMED})
        self.store.save_spec(run_id, confirmed, actor=approver)
        self.store.record_approval(
            ApprovalRecord(
                run_id=run_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=confirmed.spec_id,
                target_hash=confirmed.content_hash,
                commit_sha=commit_sha,
                approver=approver,
                decision=decision,
                reason=reason,
            )
        )
        return confirmed
