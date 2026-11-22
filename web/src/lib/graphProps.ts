import type { RunProjection } from "../api/types";
import type { DagGraphProps, DagPill, DagTone, WorkflowGraphProps } from "../graph/types";
import {
  LIVE_TASK_STATUSES,
  NODE_LABELS,
  RETRY_MESSAGE,
  RETRY_NEXT_STEP,
  statusMeta,
  taskDisplayStatus,
  workflowCurrentNode,
  WORKFLOW_STATUS_TEXT,
} from "./labels";

const TOOL_MARKS: Record<string, string> = { ok: "✓", error: "✕", denied: "⊘" };

export function dagGraphProps(
  projection: RunProjection,
  selectedId: string | null,
  onSelect: (id: string) => void,
): DagGraphProps {
  const nodes = projection.dag.nodes || [];
  const edges = projection.dag.edges || [];
  const displayStatus = new Map(nodes.map((node) => [node.task_id, taskDisplayStatus(node, projection)]));
  return {
    runId: projection.run_id,
    selectedId,
    onSelect,
    ariaLabel: "任务依赖有向无环图",
    emptyMessage: "当前运行尚未生成任务 DAG",
    nodes: nodes.map((node) => {
      const status = displayStatus.get(node.task_id) || "waiting";
      const meta = statusMeta(status, node.accepted_risk_reason);
      const dependencyBlocked = Boolean(node.blocked_by?.length);
      const retrying = Boolean(node.provider_retry);
      const live = status === "running" ? node.live_progress : null;
      const recent = live?.recent_tools.at(-1);
      const tool = live?.current_tool?.tool || (recent && `${TOOL_MARKS[recent.status]} ${recent.tool}`);
      const activity = [live?.driver_name, tool].filter(Boolean).join(" · ");
      const context = retrying ? RETRY_MESSAGE : activity || `${node.role || "engineer"} · ${node.kind || "implement"}`;
      const statusText = retrying
        ? RETRY_NEXT_STEP
        : dependencyBlocked
          ? "依赖受阻"
          : `${meta.glyph} ${meta.label}`;
      const tone: DagTone = retrying
        ? "warn"
        : dependencyBlocked
          ? "bad"
          : status === "planned"
            ? "planned"
            : meta.variant;
      const pill: DagPill | null = retrying
        ? { text: "重试中", tone: "warn" }
        : node.accepted_risk_reason
          ? { text: "⚠ 风险", tone: "warn" }
          : node.attempt > 1
            ? { text: `×${node.attempt}`, tone: "muted" }
            : null;
      return {
        id: node.task_id,
        layer: node.layer || 0,
        status,
        tone,
        meta: context,
        statusText,
        pill,
        title: node.title || "",
        issue: retrying ? "retry" : dependencyBlocked ? "dependency" : "",
        risk: Boolean(node.accepted_risk_reason),
        ariaLabel: `任务 ${node.task_id} ${retrying || activity ? `${context}，${statusText}` : statusText}`,
      };
    }),
    edges: edges.map((edge) => ({
      source: edge.from,
      target: edge.to,
      critical: Boolean(edge.critical),
      blocked: Boolean(edge.blocked_edge),
      running: !edge.blocked_edge && LIVE_TASK_STATUSES.has(displayStatus.get(edge.to) || ""),
    })),
  };
}

export function workflowGraphProps(
  projection: RunProjection,
  selectedId: string | null,
  onSelect: (id: string) => void,
): WorkflowGraphProps {
  const workflow = projection.workflow;
  const graph = workflow.graph;
  const statuses = workflow.node_statuses || {};
  const taken = new Set((workflow.traversed_edges || []).map((edge) => `${edge.source}\u2192${edge.target}`));
  return {
    runId: projection.run_id,
    selectedId,
    focusId: workflowCurrentNode(workflow),
    onSelect,
    ariaLabel: "工作流拓扑有向图",
    emptyMessage: "当前工作流未提供拓扑描述",
    nodes: (graph?.nodes || []).map((node) => {
      const status = statuses[node.name] || "pending";
      const label = NODE_LABELS[node.name] || node.name;
      const meta = statusMeta(status);
      const statusText = WORKFLOW_STATUS_TEXT[status] || `${meta.glyph} ${meta.label}`;
      return {
        id: node.name,
        label,
        kind: node.kind || "",
        status,
        statusText,
        title: `${label} · ${node.kind || node.name}`,
        ariaLabel: `${label} (${node.name}) ${statusText}`,
      };
    }),
    edges: (graph?.edges || []).map((edge) => ({
      source: edge.source,
      target: edge.target,
      conditional: Boolean(edge.conditional),
      traversed: taken.has(`${edge.source}\u2192${edge.target}`),
    })),
  };
}
