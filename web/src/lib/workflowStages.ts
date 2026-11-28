import type { RunProjection } from "../api/types";
import { localized } from "../i18n/localized";
import { tr } from "../i18n/translate";
import { NODE_LABELS, statusMeta, WORKFLOW_STATUS_TEXT } from "./labels";

type WorkflowGraph = NonNullable<RunProjection["workflow"]["graph"]>;
type WorkflowGraphNode = WorkflowGraph["nodes"][number];
type WorkflowGraphEdge = WorkflowGraph["edges"][number];

export type WorkflowNodeKind = "gate" | "validator" | "stage";
export type WorkflowStepTone = "ok" | "live" | "warn" | "bad" | "skipped" | "muted";
export type WorkflowPhaseState = "done" | "active" | "waiting" | "failed" | "pending";
export type WorkflowPhaseTone = "ok" | "live" | "warn" | "bad" | "muted";
export type ConnectorState = "traversed" | "skipped" | "pending";

export interface RouteBadge {
  text: string;
  tone: "muted" | "warn" | "bad";
}

export interface WorkflowStageStep {
  id: string;
  label: string;
  kind: WorkflowNodeKind;
  kindLabel: string;
  status: string;
  tone: WorkflowStepTone;
  glyph: string;
  statusText: string;
  routeBadge: RouteBadge | null;
  connectorBefore: ConnectorState | null;
  ariaLabel: string;
  title: string;
}

export interface WorkflowStageColumn {
  id: string;
  indexLabel: string;
  title: string;
  god: string;
  state: WorkflowPhaseState;
  statusText: string;
  statusTone: WorkflowPhaseTone;
  steps: WorkflowStageStep[];
}

export interface WorkflowStagesModel {
  columns: WorkflowStageColumn[];
  totalNodes: number;
}

interface PhaseDefinition {
  id: string;
  indexLabel: string;
  title: string;
  god: string;
}

const PHASE_DEFINITIONS: readonly PhaseDefinition[] = [
  { id: "align", indexLabel: "STAGE 01", title: "需求对齐", god: "Prometheus" },
  { id: "design", indexLabel: "STAGE 02", title: "白盒架构", god: "Athena" },
  { id: "plan", indexLabel: "STAGE 03", title: "任务编排", god: "Athena" },
  { id: "execute", indexLabel: "STAGE 04", title: "执行验收", god: "Hephaestus" },
  { id: "review", indexLabel: "STAGE 05", title: "对抗审查", god: "Momus" },
  { id: "verify", indexLabel: "STAGE 06", title: "真实验收", god: "Argus" },
  { id: "other", indexLabel: "STAGE 07", title: "扩展步骤", god: "Olympus" },
];

const PHASE_ALIASES: Record<string, string> = {
  align: "align",
  explore: "align",
  reproduce: "align",
  design: "design",
  architect: "design",
  plan: "plan",
  execute: "execute",
  build: "execute",
  hygiene: "execute",
  review: "review",
  interrogate: "review",
  verify: "verify",
  other: "other",
};

const NODE_PHASE_BY_ID = new Map<string, string>([
  ["intake", "align"],
  ["align", "align"],
  ["critic", "align"],
  ["gate_spec", "align"],
  ["design", "design"],
  ["validate_design", "design"],
  ["gate_design", "design"],
  ["plan", "plan"],
  ["validate_plan", "plan"],
  ["gate_plan", "plan"],
  ["execute", "execute"],
]);

/** Node-kind captions; proxied so every read is translated at access time. */
const KIND_LABELS: Record<WorkflowNodeKind, string> = localized<Record<WorkflowNodeKind, string>>({
  gate: "关口",
  validator: "校验",
  stage: "",
});

const STEP_GLYPHS = new Map<string, string>([
  ["completed", "✓"],
  ["running", "▶"],
  ["paused", "◷"],
  ["failed", "✕"],
  ["error", "✕"],
  ["skipped", "—"],
  ["pending", "○"],
]);

function ownValue<T>(record: Record<string, T> | null | undefined, key: string): T | undefined {
  if (!record || !Object.hasOwn(record, key)) return undefined;
  return record[key];
}

function nodeLabelFor(nodeId: string): string {
  return ownValue(NODE_LABELS, nodeId) ?? nodeId;
}

export function classifyWorkflowNode(name: string, kind?: string | null): WorkflowNodeKind {
  const raw = String(kind || "").toLowerCase();
  const id = String(name || "").toLowerCase();
  if (raw === "human_gate" || raw.startsWith("gate") || id.startsWith("gate_")) return "gate";
  if (raw.startsWith("validate") || id.startsWith("validate_")) return "validator";
  return "stage";
}

export function workflowStepTone(status: string): WorkflowStepTone {
  switch (status) {
    case "completed":
      return "ok";
    case "running":
      return "live";
    case "paused":
      return "warn";
    case "failed":
    case "error":
      return "bad";
    case "skipped":
      return "skipped";
    default:
      return "muted";
  }
}

function resolvePhaseId(node: WorkflowGraphNode): string {
  const declaredPhase = typeof node.phase === "string" ? node.phase.trim().toLowerCase() : "";
  if (declaredPhase) {
    return ownValue(PHASE_ALIASES, declaredPhase) ?? "other";
  }
  const direct = NODE_PHASE_BY_ID.get(node.name);
  if (direct) return direct;
  const id = node.name.toLowerCase();
  const kind = String(node.kind || "").toLowerCase();
  if (id.includes("design") || kind.includes("design")) return "design";
  if (id.includes("plan") || kind.includes("plan") || kind.includes("decompose")) return "plan";
  if (id.includes("review") || id.includes("interrogate") || kind.includes("interrogate")) {
    return "review";
  }
  if (id.includes("exec") || id.includes("verify") || kind.includes("exec") || kind.includes("verify")) {
    return "execute";
  }
  if (
    id.includes("intake") ||
    id.includes("align") ||
    id.includes("critic") ||
    id.includes("spec") ||
    kind.includes("align") ||
    kind.includes("intake")
  ) {
    return "align";
  }
  return "other";
}

function buildRouteBadge(
  nodeId: string,
  status: string,
  nodePhase: string,
  displayOrder: Map<string, number>,
  nodePhaseMap: Map<string, string>,
  kindMap: Map<string, WorkflowNodeKind>,
  edges: WorkflowGraphEdge[],
  traversedSet: Set<string>,
): RouteBadge | null {
  const srcIndex = displayOrder.get(nodeId);
  if (srcIndex === undefined) return null;

  const outgoing = edges.filter((edge) => edge.source === nodeId && displayOrder.has(edge.target));

  let bestTraversedBack: { target: string; index: number } | null = null;
  let bestTraversedForward: { target: string; index: number } | null = null;
  let bestDeclaredBack: { target: string; index: number } | null = null;

  for (const edge of outgoing) {
    const dstIndex = displayOrder.get(edge.target)!;
    const isTraversed = traversedSet.has(`${edge.source}\u2192${edge.target}`);
    if (isTraversed) {
      if (dstIndex <= srcIndex) {
        if (!bestTraversedBack || dstIndex < bestTraversedBack.index) {
          bestTraversedBack = { target: edge.target, index: dstIndex };
        }
      } else if (dstIndex > srcIndex + 1) {
        if (!bestTraversedForward || dstIndex > bestTraversedForward.index) {
          bestTraversedForward = { target: edge.target, index: dstIndex };
        }
      }
    } else if (dstIndex <= srcIndex) {
      if (!bestDeclaredBack || dstIndex < bestDeclaredBack.index) {
        bestDeclaredBack = { target: edge.target, index: dstIndex };
      }
    }
  }

  if (bestTraversedBack) {
    return { text: tr("回退{node}", { node: nodeLabelFor(bestTraversedBack.target) }), tone: "warn" };
  }

  if (bestTraversedForward) {
    const target = bestTraversedForward.target;
    const targetKind = kindMap.get(target);
    const samePhase = nodePhaseMap.get(target) === nodePhase;
    if (samePhase && targetKind === "gate") {
      return { text: tr("直达关口"), tone: "muted" };
    }
    return { text: tr("直达{node}", { node: nodeLabelFor(target) }), tone: "muted" };
  }

  if ((status === "failed" || status === "error") && bestDeclaredBack) {
    return { text: tr("回退 {node}", { node: nodeLabelFor(bestDeclaredBack.target) }), tone: "bad" };
  }

  return null;
}

function summarizePhase(steps: WorkflowStageStep[]): {
  state: WorkflowPhaseState;
  statusText: string;
  statusTone: WorkflowPhaseTone;
} {
  const total = steps.length;
  const completed = steps.filter((step) => step.status === "completed").length;
  const skipped = steps.filter((step) => step.status === "skipped").length;

  if (steps.some((step) => step.status === "failed" || step.status === "error")) {
    return { state: "failed", statusText: tr("✕ 中断"), statusTone: "bad" };
  }
  if (steps.some((step) => step.status === "paused")) {
    return { state: "waiting", statusText: tr("◷ 待确认"), statusTone: "warn" };
  }
  if (steps.some((step) => step.status === "running")) {
    return { state: "active", statusText: tr("▶ 运行中"), statusTone: "live" };
  }
  if (completed + skipped === total && total > 0) {
    if (completed === 0) {
      return { state: "done", statusText: tr("— 已跳过"), statusTone: "muted" };
    }
    return { state: "done", statusText: `✓ ${completed}/${total}`, statusTone: "ok" };
  }
  if (completed > 0) {
    return { state: "active", statusText: `▶ ${completed}/${total}`, statusTone: "live" };
  }
  return { state: "pending", statusText: tr("◷ 待开始"), statusTone: "muted" };
}

function connectorBetween(
  prev: WorkflowStageStep,
  cur: WorkflowStageStep,
  traversedSet: Set<string>,
): ConnectorState {
  if (traversedSet.has(`${prev.id}\u2192${cur.id}`)) return "traversed";
  if (prev.status === "skipped" || cur.status === "skipped") return "skipped";
  return "pending";
}

export function buildWorkflowStages(projection: RunProjection): WorkflowStagesModel {
  const workflow = projection.workflow;
  const rawNodes = workflow.graph?.nodes || [];
  const rawEdges = workflow.graph?.edges || [];
  const statuses = workflow.node_statuses || {};

  const seenNodes = new Set<string>();
  const uniqueNodes: WorkflowGraphNode[] = [];
  for (const node of rawNodes) {
    if (!node?.name || seenNodes.has(node.name)) continue;
    seenNodes.add(node.name);
    uniqueNodes.push(node);
  }

  if (!uniqueNodes.length) {
    return { columns: [], totalNodes: 0 };
  }

  const traversedSet = new Set(
    (workflow.traversed_edges || []).map((edge) => `${edge.source}\u2192${edge.target}`),
  );

  const nodePhaseMap = new Map<string, string>();
  const kindMap = new Map<string, WorkflowNodeKind>();
  const grouped = new Map<string, WorkflowGraphNode[]>();

  for (const node of uniqueNodes) {
    const phaseId = resolvePhaseId(node);
    nodePhaseMap.set(node.name, phaseId);
    kindMap.set(node.name, classifyWorkflowNode(node.name, node.kind));
    const list = grouped.get(phaseId);
    if (list) list.push(node);
    else grouped.set(phaseId, [node]);
  }

  const displayOrder = new Map<string, number>();
  for (const def of PHASE_DEFINITIONS) {
    for (const node of grouped.get(def.id) || []) {
      displayOrder.set(node.name, displayOrder.size);
    }
  }

  // When graph.start is declared, DFS reverse-postorder yields a true topological ordering on
  // any DAG (preventing convergence shortcuts like align -> gate_spec or custom review -> execute
  // from being mislabeled as back-edges) while still detecting genuine cycle-closing edges.
  // When graph.start is omitted (partial/synthetic graph payloads), fall back to displayOrder.
  const startNodes = (workflow.graph?.start || []).filter((name) => seenNodes.has(name));
  let orderMap: Map<string, number> = displayOrder;
  if (startNodes.length > 0) {
    const declaredIndex = new Map<string, number>();
    for (let i = 0; i < uniqueNodes.length; i += 1) {
      declaredIndex.set(uniqueNodes[i].name, i);
    }
    const adjacency = new Map<string, { target: string; conditional: boolean }[]>();
    for (const edge of rawEdges) {
      if (!seenNodes.has(edge.source) || !seenNodes.has(edge.target)) continue;
      const entry = { target: edge.target, conditional: Boolean(edge.conditional) };
      const targets = adjacency.get(edge.source);
      if (targets) targets.push(entry);
      else adjacency.set(edge.source, [entry]);
    }
    for (const targets of adjacency.values()) {
      targets.sort((a, b) => {
        if (a.conditional !== b.conditional) return a.conditional ? 1 : -1;
        return (declaredIndex.get(a.target) ?? 0) - (declaredIndex.get(b.target) ?? 0);
      });
    }
    const visitState = new Map<string, number>(); // 1 = visiting (on stack), 2 = finished
    const postOrder: string[] = [];
    const visit = (nodeName: string): void => {
      visitState.set(nodeName, 1);
      for (const edge of adjacency.get(nodeName) || []) {
        if (visitState.get(edge.target) === undefined) {
          visit(edge.target);
        }
      }
      visitState.set(nodeName, 2);
      postOrder.push(nodeName);
    };
    for (const startNode of startNodes) {
      if (visitState.get(startNode) === undefined) {
        visit(startNode);
      }
    }
    for (const [nodeName] of displayOrder) {
      if (visitState.get(nodeName) === undefined) {
        visit(nodeName);
      }
    }
    postOrder.reverse();
    orderMap = new Map<string, number>();
    for (const nodeName of postOrder) {
      orderMap.set(nodeName, orderMap.size);
    }
  }

  const columns: WorkflowStageColumn[] = [];
  for (const def of PHASE_DEFINITIONS) {
    const nodesInPhase = grouped.get(def.id);
    if (!nodesInPhase?.length) continue;

    const steps: WorkflowStageStep[] = nodesInPhase.map((node) => {
      const status = ownValue(statuses, node.name) ?? "pending";
      const label = nodeLabelFor(node.name);
      const kind = kindMap.get(node.name) || "stage";
      const kindLabel = KIND_LABELS[kind];
      const meta = statusMeta(status);
      const statusText = ownValue(WORKFLOW_STATUS_TEXT, status) ?? `${meta.glyph} ${meta.label}`;
      const routeBadge = buildRouteBadge(
        node.name,
        status,
        def.id,
        orderMap,
        nodePhaseMap,
        kindMap,
        rawEdges,
        traversedSet,
      );
      const baseAria = `${label} (${node.name}) ${statusText}`;
      return {
        id: node.name,
        label,
        kind,
        kindLabel,
        status,
        tone: workflowStepTone(status),
        glyph: STEP_GLYPHS.get(status) ?? "○",
        statusText,
        routeBadge,
        connectorBefore: null,
        ariaLabel: kindLabel ? `${baseAria} · ${kindLabel}` : baseAria,
        title: `${label} · ${node.kind || node.name}`,
      };
    });

    for (let i = 1; i < steps.length; i += 1) {
      steps[i] = {
        ...steps[i],
        connectorBefore: connectorBetween(steps[i - 1], steps[i], traversedSet),
      };
    }

    const summary = summarizePhase(steps);
    columns.push({
      id: def.id,
      indexLabel: def.indexLabel,
      title: tr(def.title),
      god: def.god,
      ...summary,
      steps,
    });
  }

  return { columns, totalNodes: uniqueNodes.length };
}
