import type { DagNode, RunProgressItem, RunProjection, WorkflowView } from "../api/types";

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

export const MATURITY_LABELS: Record<string, string> = {
  raw_intent: "原始意图",
  goal_spec_ready: "GoalSpec 就绪",
  design_spec_ready: "DesignSpec 就绪",
};

export const RETRY_MESSAGE = "调用超时，正在自动重试";
export const RETRY_NEXT_STEP = "等待本次调用返回";

export const WORKFLOW_LABELS: Record<string, string> = {
  product_blueprint: "产品蓝图",
  full_sdlc_tdd: "全流程 TDD",
  repo_fast_patch: "快速补丁",
};

export const WORKFLOW_SELECT_LABELS: Record<string, string> = {
  product_blueprint: "product_blueprint · 产品蓝图（仅规划，不写代码）",
  full_sdlc_tdd: "full_sdlc_tdd · 全流程研发（含沙盒写代码 + 验收）",
  repo_fast_patch: "repo_fast_patch · 快速补丁（跳过架构直改代码）",
};

export const WORKFLOW_SUMMARY: Record<string, string> = {
  product_blueprint:
    "需求对齐 → 白盒架构 → Wave 0/1/2 任务拆解后结束；含 3 道人工关口，不写仓库代码。",
  full_sdlc_tdd:
    "需求对齐 → 白盒架构 → 任务拆解 → 隔离沙盒并发写代码与冷启动验收；含 3 道人工关口。",
  repo_fast_patch: "跳过白盒架构，基于 GoalSpec 拆解后直接改代码并验收；人工关口全自动放行。",
};

export const NODE_LABELS: Record<string, string> = {
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
};

export const WORKFLOW_STATUS_TEXT: Record<string, string> = {
  completed: "✓ 已完成",
  running: "▶ 运行中",
  paused: "◷ 待确认",
  failed: "✕ 中断/阻塞",
  error: "✕ 中断/阻塞",
  skipped: "— 已跳过",
  pending: "◷ 等待",
};

export const APPROVAL_LABELS = {
  approve_spec: "GoalSpec",
  approve_design: "DesignSpec",
  approve_plan: "任务计划",
} as const;

export const APPROVAL_BUTTONS = {
  approve_spec: { id: "btn-approve-spec", label: "批准 GoalSpec 进入架构设计" },
  approve_design: { id: "btn-approve-design", label: "批准 DesignSpec 启动任务分解" },
  approve_plan: { id: "btn-approve-plan", label: "批准任务计划" },
} as const;

export const OUTPUT_ACTIONS: Record<string, string> = {
  spec: "查看 GoalSpec",
  design: "查看 DesignSpec",
  plan: "查看任务依赖图",
  execution: "查看代码变更",
  verification: "查看验收证据",
};

export const DELIVERY_STATUS: Record<string, [string, Variant]> = {
  passed: ["✓ 通过", "ok"],
  failed: ["✕ 失败", "bad"],
  unverified: ["◷ 未验证", "muted"],
};

export const LIVE_TASK_STATUSES = new Set(["running", "verifying"]);

export function localizeTriageReason(text: string): string {
  const raw = String(text || "").trim();
  if (raw.startsWith("Free-form intent without frozen requirements")) {
    return "输入为自然语言描述，尚未冻结可验收需求、模块边界或接口契约。";
  }
  if (raw.startsWith("Stage 1 Align must synthesize an EARS GoalSpec")) {
    return "需先经 Stage 1 需求对齐生成 EARS GoalSpec，再进入白盒架构设计。";
  }
  if (raw.startsWith("Input describes WHAT the product must do")) {
    return "已明确业务目标（What），待 Stage 2a 补齐模块路径、数据归属与接口接缝（How）。";
  }
  if (raw.startsWith("No interface contracts")) {
    return "缺少明确的模块间接口签名与异常分支契约。";
  }
  if (raw.startsWith("No data ownership map")) {
    return "缺少数据实体及其唯一所属模块的归属映射。";
  }
  if (raw.startsWith("GoalSpec passed deterministic validation")) {
    return "输入已达到 GoalSpec 就绪标准，可直接跨过澄清进入白盒架构设计。";
  }
  if (raw.startsWith("DesignSpec passed deterministic validation")) {
    return "输入已达到完整 DesignSpec 标准，可直接进入任务分解（PM Decompose）。";
  }
  return raw;
}

export function localizeError(text: string | null | undefined): string {
  const raw = String(text || "").trim();
  if (!raw) return "";
  if (/read operation timed out|ReadTimeout/i.test(raw)) {
    return `模型接口读取超时（已将默认超时阈值提升至 180s，可直接点击“从断点重试”继续）· ${raw}`;
  }
  if (/provider_error/i.test(raw)) {
    return `模型调用异常（可检查提供方配置后点击“从断点重试”）· ${raw}`;
  }
  return raw;
}

export function statusMeta(status: string | null | undefined, riskReason?: string | null): StatusMeta {
  if (riskReason) return { label: "未验证 · 人工接受", variant: "warn", glyph: "⚠" };
  switch (status) {
    case "accepted":
    case "verified":
    case "done":
      return { label: "已验收", variant: "ok", glyph: "✓" };
    case "completed":
      return { label: "已完成", variant: "ok", glyph: "✓" };
    case "running":
      return { label: "运行中", variant: "live", glyph: "▶" };
    case "aligning":
      return { label: "对齐中", variant: "live", glyph: "▶" };
    case "designing":
      return { label: "设计中", variant: "live", glyph: "▶" };
    case "planning":
      return { label: "分解中", variant: "live", glyph: "▶" };
    case "awaiting_human":
      return { label: "待人工", variant: "warn", glyph: "⚠" };
    case "created":
      return { label: "已创建", variant: "muted", glyph: "◷" };
    case "verifying":
      return { label: "待验收", variant: "live", glyph: "▶" };
    case "blocked":
      return { label: "阻塞", variant: "bad", glyph: "✕" };
    case "failed":
    case "error":
      return { label: "失败", variant: "bad", glyph: "✕" };
    case "paused":
      return { label: "已暂停", variant: "warn", glyph: "◷" };
    case "ready":
      return { label: "就绪", variant: "muted", glyph: "◷" };
    case "cancelled":
      return { label: "已取消", variant: "muted", glyph: "◷" };
    case "planned":
      return { label: "蓝图已定稿", variant: "muted", glyph: "✓" };
    default:
      return { label: "等待", variant: "muted", glyph: "◷" };
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

export function progressText(event: RunProgressItem | null | undefined): string {
  if (!event) return "等待首条运行记录";
  if (event.summary) return event.summary;
  const node = (event.node && NODE_LABELS[event.node]) || event.node || "工作流";
  const labels: Record<string, string> = {
    "run.created": "已接收编排请求",
    "intake.triaged": "输入分流完成",
    "spec.clarified": "已记录澄清回答",
    "approval.recorded": "已记录人工决定",
    "graph.node_started": `开始${node}`,
    "graph.node_finished": `${node}完成`,
    "graph.paused": `${node}等待继续`,
    "graph.failed": `${node}执行中断`,
    "graph.finished": "工作流完成",
    "run.status_changed": `运行状态更新为${statusMeta(event.to).label}`,
    "run.workflow_promoted": "已切换为全流程执行",
    "task.requeued": `${event.task_id} 已重新排队`,
    "task.blocked": `${event.task_id} 受阻`,
    "task.accepted": `${event.task_id} 已验收`,
    "task.accepted_with_risk": `${event.task_id} 保留未验证状态，已人工接受风险`,
  };
  return labels[event.event_type] || "已记录运行更新";
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
      ? `${name} · 有任务受阻`
      : projection.status === "failed"
        ? `${name} · 执行中断`
        : name;
  return { key: node || stage?.id || projection.status, label };
}

export function unresolvedAmbiguities(spec: { open_ambiguities?: { blocking: boolean; answer?: string | null }[] } | null | undefined) {
  return (spec?.open_ambiguities || []).filter((amb) => amb && amb.blocking && !amb.answer);
}

export function pendingAmbiguities(projection: RunProjection) {
  return (projection.spec?.open_ambiguities || []).filter((amb) => amb && !amb.answer);
}

/** The task whose blocker drives the banner: the selected task's own blocker, else the banner. */
export function taskBlocker(projection: RunProjection | null, selectedTaskId: string | null) {
  const banner = projection?.hitl_banner;
  if (!projection || banner?.kind !== "task_blocked") return null;
  const selected = projection.dag.nodes.find((node: DagNode) => node.task_id === selectedTaskId);
  return selected?.blocker || banner;
}
