import type {
  BallHolder,
  DagNode,
  PantheonGod,
  PendingAction,
  RunProgressItem,
  RunProjection,
  TaskType,
  WorkflowView,
  WorkspaceMode,
} from "../api/types";
import { localized } from "../i18n/localized";
import { tr } from "../i18n/translate";

export const VIEWS = ["topology", "design", "matrix", "waterfall"] as const;
export type View = (typeof VIEWS)[number];

export const TABS = ["overview", "artifacts", "events"] as const;
export type Tab = (typeof TABS)[number];

export const isView = (value: unknown): value is View => VIEWS.includes(value as View);
export const isTab = (value: unknown): value is Tab => TABS.includes(value as Tab);

export type Variant = "ok" | "live" | "warn" | "bad" | "muted";

export interface StatusMeta {
  label: string;
  variant: Variant;
  glyph: string;
}

/**
 * Copy tables in this module are wrapped with `localized()`, so every read is
 * translated at access time — call sites keep using `TABLE[key]` unchanged.
 */
export const MATURITY_LABELS = localized<Record<string, string>>({
  raw_intent: "原始意图",
  goal_spec_ready: "GoalSpec 就绪",
  design_spec_ready: "DesignSpec 就绪",
});

export const retryMessage = (): string => tr("调用超时，正在自动重试");
export const retryNextStep = (): string => tr("等待本次调用返回");

export const WORKFLOW_LABELS = localized<Record<string, string>>({
  product_blueprint: "产品蓝图",
  full_sdlc_tdd: "全流程 TDD",
  repo_fast_patch: "快速补丁",
});

export const WORKFLOW_SELECT_LABELS = localized<Record<string, string>>({
  product_blueprint: "product_blueprint · 产品蓝图（仅规划，不写代码）",
  full_sdlc_tdd: "full_sdlc_tdd · 全流程研发（含沙盒写代码 + 验收）",
  repo_fast_patch: "repo_fast_patch · 快速补丁（跳过架构直改代码）",
});

export const WORKFLOW_SUMMARY = localized<Record<string, string>>({
  product_blueprint:
    "需求对齐 → 白盒架构 → Wave 0/1/2 任务拆解后结束；含 3 道人工关口，不写仓库代码。",
  full_sdlc_tdd:
    "需求对齐 → 白盒架构 → 任务拆解 → 隔离沙盒并发写代码与冷启动验收；含 3 道人工关口。",
  repo_fast_patch: "跳过白盒架构，基于 GoalSpec 拆解后直接改代码并验收；人工关口全自动放行。",
});

export const NODE_LABELS = localized<Record<string, string>>({
  intake: "输入分流",
  align: "需求对齐",
  critic: "多维评审",
  gate_spec: "需求确认",
  design: "白盒架构",
  validate_design: "架构校验",
  gate_design: "架构确认",
  plan: "任务分解",
  validate_plan: "计划校验",
  gate_plan: "计划确认",
  execute: "并发执行",
});

export const WORKFLOW_STATUS_TEXT = localized<Record<string, string>>({
  completed: "✓ 已完成",
  running: "▶ 运行中",
  paused: "◷ 待确认",
  failed: "✕ 中断/阻塞",
  error: "✕ 中断/阻塞",
  skipped: "— 已跳过",
  pending: "◷ 等待",
});

export const APPROVAL_LABELS = localized({
  approve_spec: "GoalSpec",
  approve_design: "DesignSpec",
  approve_plan: "任务计划",
} as const);

export const APPROVAL_BUTTONS = localized({
  approve_spec: { id: "btn-approve-spec", label: "批准 GoalSpec 进入架构设计" },
  approve_design: { id: "btn-approve-design", label: "批准 DesignSpec 启动任务分解" },
  approve_plan: { id: "btn-approve-plan", label: "批准任务计划" },
} as const);

export const OUTPUT_ACTIONS = localized<Record<string, string>>({
  spec: "查看 GoalSpec",
  design: "查看 DesignSpec",
  plan: "查看任务依赖图",
  execution: "查看代码变更",
  verification: "查看验收证据",
});

export const DELIVERY_STATUS = localized<Record<string, [string, Variant]>>({
  passed: ["✓ 通过", "ok"],
  failed: ["✕ 失败", "bad"],
  unverified: ["◷ 未验证", "muted"],
});

export const POST_DAG_STATUS = localized<Record<string, [string, Variant]>>({
  idle: ["◷ 待验证", "muted"],
  verifying: ["▶ 验证中", "live"],
  repairing: ["▶ 定向修复中", "live"],
  passed: ["✓ 交付已通过", "ok"],
  paused: ["◷ 已暂停", "warn"],
  failed: ["✕ 交付未通过", "bad"],
  awaiting_human: ["⚠ 待人工裁决", "warn"],
});

export const POST_DAG_PHASE = localized<Record<string, string>>({
  verify: "实机验证",
  recheck: "复现复检",
  review: "Argus 复核",
  repair: "定向修复",
});

export const POST_DAG_OUTCOME = localized<Record<string, [string, Variant]>>({
  passed: ["✓ 通过", "ok"],
  applied: ["✓ 已应用", "ok"],
  completed: ["✓ 已完成", "ok"],
  failed: ["✕ 失败", "bad"],
  rejected: ["✕ 已拒绝", "bad"],
});

export function formatDurationMs(durationMs: number | null | undefined): string {
  const value = Number(durationMs) || 0;
  if (value <= 0) return tr("未记录耗时");
  if (value >= 1000) return `${(value / 1000).toFixed(1)} s`;
  return `${value} ms`;
}

export const LIVE_TASK_STATUSES = new Set(["running", "verifying"]);

export function localizeTriageReason(text: string): string {
  const raw = String(text || "").trim();
  if (raw.startsWith("Free-form intent without frozen requirements")) {
    return tr("输入为自然语言描述，尚未冻结可验收需求、模块边界或接口契约。");
  }
  if (raw.startsWith("Stage 1 Align must synthesize an EARS GoalSpec")) {
    return tr("需先经 Stage 1 需求对齐生成 EARS GoalSpec，再进入白盒架构设计。");
  }
  if (raw.startsWith("Input describes WHAT the product must do")) {
    return tr("已明确业务目标（What），待 Stage 2a 补齐模块路径、数据归属与接口接缝（How）。");
  }
  if (raw.startsWith("No interface contracts")) {
    return tr("缺少明确的模块间接口签名与异常分支契约。");
  }
  if (raw.startsWith("No data ownership map")) {
    return tr("缺少数据实体及其唯一所属模块的归属映射。");
  }
  if (raw.startsWith("GoalSpec passed deterministic validation")) {
    return tr("输入已达到 GoalSpec 就绪标准，可直接跨过澄清进入白盒架构设计。");
  }
  if (raw.startsWith("DesignSpec passed deterministic validation")) {
    return tr("输入已达到完整 DesignSpec 标准，可直接进入任务分解（PM Decompose）。");
  }
  return raw;
}

export function localizeError(text: string | null | undefined): string {
  const raw = String(text || "").trim();
  if (!raw) return "";
  if (/read operation timed out|ReadTimeout/i.test(raw)) {
    return tr(
      "模型接口读取超时（已将默认超时阈值提升至 300s，可直接点击“从断点重试”继续）· {raw}",
      { raw },
    );
  }
  if (/provider_error/i.test(raw)) {
    return tr("模型调用异常（可检查提供方配置后点击“从断点重试”）· {raw}", { raw });
  }
  return raw;
}

export function statusMeta(status: string | null | undefined, riskReason?: string | null): StatusMeta {
  if (riskReason) return { label: tr("未验证 · 人工接受"), variant: "warn", glyph: "⚠" };
  switch (status) {
    case "accepted":
    case "verified":
    case "done":
      return { label: tr("已验收"), variant: "ok", glyph: "✓" };
    case "completed":
      return { label: tr("已完成"), variant: "ok", glyph: "✓" };
    case "running":
      return { label: tr("运行中"), variant: "live", glyph: "▶" };
    case "aligning":
      return { label: tr("对齐中"), variant: "live", glyph: "▶" };
    case "designing":
      return { label: tr("设计中"), variant: "live", glyph: "▶" };
    case "planning":
      return { label: tr("分解中"), variant: "live", glyph: "▶" };
    case "awaiting_human":
      return { label: tr("待人工"), variant: "warn", glyph: "⚠" };
    case "created":
      return { label: tr("已创建"), variant: "muted", glyph: "◷" };
    case "verifying":
      return { label: tr("待验收"), variant: "live", glyph: "▶" };
    case "blocked":
      return { label: tr("阻塞"), variant: "bad", glyph: "✕" };
    case "failed":
    case "error":
      return { label: tr("失败"), variant: "bad", glyph: "✕" };
    case "paused":
      return { label: tr("已暂停"), variant: "warn", glyph: "◷" };
    case "ready":
      return { label: tr("就绪"), variant: "muted", glyph: "◷" };
    case "cancelled":
      return { label: tr("已取消"), variant: "muted", glyph: "◷" };
    case "planned":
      return { label: tr("蓝图已定稿"), variant: "muted", glyph: "✓" };
    default:
      return { label: tr("等待"), variant: "muted", glyph: "◷" };
  }
}

export function taskDisplayStatus(
  task: { status?: string | null } | null | undefined,
  projection: RunProjection | null,
): string {
  if (projection?.blueprint_completed && task?.status === "waiting") return "planned";
  return task?.status || "waiting";
}

export function formatTokens(value: number | null | undefined): string {
  const count = Number(value) || 0;
  return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count);
}

export function workflowCurrentNode(workflow: Partial<WorkflowView>): string | null {
  return workflow.current_node || workflow.failed_node || workflow.paused?.node || null;
}

const PROGRESS_LABELS = localized<Record<string, string>>({
  "run.created": "已接收编排请求",
  "intake.triaged": "输入分流完成",
  "spec.clarified": "已记录澄清回答",
  "approval.recorded": "已记录人工决定",
  "graph.node_started": "开始{node}",
  "graph.node_finished": "{node}完成",
  "graph.paused": "{node}等待继续",
  "graph.failed": "{node}执行中断",
  "graph.finished": "工作流完成",
  "run.status_changed": "运行状态更新为{status}",
  "run.workflow_promoted": "已切换为全流程执行",
  "task.requeued": "{task_id} 已重新排队",
  "task.blocked": "{task_id} 受阻",
  "task.accepted": "{task_id} 已验收",
  "task.accepted_with_risk": "{task_id} 保留未验证状态，已人工接受风险",
  "spec.clarification_note": "已补充澄清说明",
  "spec.resynthesized": "已按澄清重新合成需求规格",
  "spec.resynthesis_failed": "需求融合失败，可原地重试",
  "run.revision_requested": "已请求修订当前交付",
  "run.reopened": "运行已重新打开",
  "task.ephemeral_artifacts_cleaned": "{task_id} 临时产物已清理",
  "task.diff_gate_rejected": "{task_id} 变更越界被拦截",
  "task.escape_hatch_rejected": "{task_id} 逃逸语句被拦截",
  "task.review_deferred": "{task_id} 独立复核延后至交付验收",
  "task.deferred_review_passed": "{task_id} 延后复核已通过",
  "review.verdict_verified": "{task_id} 审查结论已核验",
  "review.repro_rechecked": "{task_id} 复现命令已复检",
  "review.repro_commands_cleared": "{task_id} 复现命令已清除",
  "product.verified": "产品端到端验证完成",
  "post_dag.repro_rechecked": "交付后复现命令已复检",
  "post_dag.repair_turn_completed": "交付修复轮次完成",
  "post_dag.repair_applied": "交付修复已应用",
  "post_dag.repair_rejected": "交付修复被拒绝",
});

export function progressText(event: RunProgressItem | null | undefined): string {
  if (!event) return tr("等待首条运行记录");
  if (event.summary) return event.summary;
  const node = (event.node && NODE_LABELS[event.node]) || event.node || tr("工作流");
  const template = PROGRESS_LABELS[event.event_type] || "已记录运行更新";
  return tr(template, {
    node,
    task_id: event.task_id || "",
    status: statusMeta(event.to).label,
  });
}

export interface CurrentStage {
  key: string;
  label: string;
}

export function currentStage(projection: RunProjection): CurrentStage {
  const workflow = projection.workflow;
  if (projection.status === "completed" || projection.status === "cancelled") {
    return { key: projection.status, label: statusMeta(projection.status).label };
  }
  const node = workflow.current_node || workflow.paused?.node || workflow.failed_node;
  const stages = projection.stage_pipeline || [];
  const stage =
    stages.find(({ status }) => status === "blocked") ||
    stages.findLast(({ status }) => status === "active");
  const name = (node && NODE_LABELS[node]) || node || stage?.label || statusMeta(projection.status).label;
  const label =
    projection.hitl_banner?.kind === "task_blocked"
      ? `${name}${tr(" · 有任务受阻")}`
      : projection.status === "failed"
        ? `${name}${tr(" · 执行中断")}`
        : name;
  return { key: node || stage?.id || projection.status, label };
}

export function unresolvedAmbiguities(spec: { open_ambiguities?: { blocking: boolean; answer?: string | null }[] } | null | undefined) {
  return (spec?.open_ambiguities || []).filter((amb) => amb && amb.blocking && !amb.answer);
}

export function isAlignClarificationActive(projection: RunProjection | null | undefined): boolean {
  return Boolean(
    projection &&
      !projection.spec_approved &&
      projection.status !== "completed" &&
      projection.status !== "cancelled",
  );
}

export function pendingAmbiguities(projection: RunProjection) {
  if (!isAlignClarificationActive(projection)) return [];
  return (projection.spec?.open_ambiguities || []).filter((amb) => amb && !amb.answer);
}

export function clarifyPanelVisibility(projection: RunProjection | null | undefined) {
  const isTerminal =
    projection?.status === "completed" || projection?.status === "cancelled";
  const inAlign = isAlignClarificationActive(projection);
  const pending = projection ? pendingAmbiguities(projection) : [];
  const submittedNotes = projection?.clarification_notes || [];
  const hasUnconsumedNotes = !isTerminal && submittedNotes.some((note) => !note.consumed);
  const pausedCraftClarify = Boolean(
    inAlign &&
      !projection?.spec &&
      (projection?.status === "awaiting_human" || projection?.status === "paused") &&
      projection?.workflow?.paused?.node &&
      projection?.workflow?.family !== "sdlc",
  );
  const showForm =
    pending.length > 0 ||
    pausedCraftClarify ||
    Boolean(inAlign && projection?.spec && projection.resynthesis_error);
  const showNotes = submittedNotes.length > 0 && (inAlign || hasUnconsumedNotes);
  return {
    pending,
    submittedNotes,
    showForm,
    showNotes,
    showPanel: showForm || showNotes,
  };
}

/** The task whose blocker drives the banner: the selected task's own blocker, else the banner. */
export function taskBlocker(projection: RunProjection | null, selectedTaskId: string | null) {
  const banner = projection?.hitl_banner;
  if (!projection || banner?.kind !== "task_blocked") return null;
  const selected = projection.dag.nodes.find((node: DagNode) => node.task_id === selectedTaskId);
  return selected?.blocker || banner;
}

export const WORKSPACE_MODE_LABELS = localized<Record<WorkspaceMode, string>>({
  brownfield: "Brownfield",
  greenfield: "Greenfield",
  spec_only: "Spec-Only",
});

export const TASK_TYPE_LABELS = localized<Record<TaskType, string>>({
  feature: "Feature",
  fix: "Fix",
  refactor: "Refactor",
  blueprint: "Blueprint",
});

export const GOD_META = localized<Record<PantheonGod, { name: string; role: string; monogram: string }>>({
  prometheus: { name: "Prometheus", role: "Align · Needs", monogram: "P" },
  athena: { name: "Athena", role: "Design · Arch", monogram: "A" },
  hephaestus: { name: "Hephaestus", role: "Build · Sandbox", monogram: "H" },
  momus: { name: "Momus", role: "Review · Audit", monogram: "M" },
  argus: { name: "Argus", role: "Verify · Gate", monogram: "G" },
});

export const STATUS_BADGE_LABELS = localized<Record<string, string>>({
  clarify: "Clarify",
  approve_spec: "Approve Spec",
  approve_design: "Approve Design",
  approve_plan: "Approve Plan",
  review_delivery: "Review Delivery",
  blocked: "Blocked",
  paused: "Paused",
  failed: "Failed",
  created: "Queued",
  aligning: "Aligning",
  designing: "Designing",
  planning: "Planning",
  building: "Building",
  reviewing: "Reviewing",
  verifying: "Verifying",
  delivered: "Delivered",
  blueprint: "Blueprint Ready",
  cancelled: "Cancelled",
});

export const PENDING_ACTION_CTA = localized<Record<NonNullable<PendingAction>, string>>({
  clarify: "Answer in Drawer",
  approve_spec: "Review Spec",
  approve_design: "Approve Design",
  approve_plan: "Approve Plan",
  resolve_blocker: "Resolve Blocker",
  review_delivery: "Review Delivery",
});

export function boardBadgeVariant(ballHolder: BallHolder, statusBadge: string): Variant {
  if (statusBadge === "failed" || statusBadge === "blocked") return "bad";
  if (statusBadge === "cancelled") return "muted";
  if (ballHolder === "human") return "warn";
  if (ballHolder === "agent") return "live";
  return "ok";
}

export function shortOracleCode(runId: string): string {
  const cleaned = runId.replace(/^run[-_]?/i, "");
  return cleaned.length > 12 ? cleaned.slice(0, 12) : cleaned || runId;
}

/**
 * Some strings arrive from the backend already composed around a Chinese prefix
 * (`projection.py`): "人工接受风险：{reason}", "前置任务待完成：{ids}", …
 * Split the prefix off so the fixed part can be localized while the payload is
 * preserved verbatim; unknown sentences fall through unchanged.
 */
export function localizeEvidenceNote(text: string | null | undefined): string {
  const raw = String(text ?? "");
  if (!raw) return "";
  const risk = raw.match(/^人工接受风险：(.+)$/);
  if (risk) return tr("人工接受风险：{reason}", { reason: risk[1] });
  const pendingDeps = raw.match(/^前置任务待完成：(.+)$/);
  if (pendingDeps) return tr("前置任务待完成：{deps}", { deps: pendingDeps[1] });
  const failedTask = raw.match(/^(.+) 验证未通过$/);
  if (failedTask) return tr("{taskId} 验证未通过", { taskId: failedTask[1] });
  const exitCode = raw.match(/^验收命令退出码 (\d+)，详见执行日志$/);
  if (exitCode) return tr("验收命令退出码 {code}，详见执行日志", { code: exitCode[1] });
  return tr(raw);
}

/**
 * Localize a backend string that glues a known label to a raw id with " · "
 * (e.g. "需求对齐 · spec-17ab"). Each segment is translated on its own, unknown
 * segments pass through untouched.
 */
export function localizeComposite(text: string | null | undefined): string {
  const raw = String(text ?? "");
  if (!raw) return "";
  if (!raw.includes(" · ")) return tr(raw);
  return raw
    .split(" · ")
    .map((part) => tr(part))
    .join(" · ");
}
