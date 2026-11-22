import type { ElkExtendedEdge, ElkNode, ElkPort } from "elkjs/lib/elk-api";

import type { Point } from "../geometry";
import {
  CARD_H,
  CARD_W,
  cardHeights,
  edgeKey,
  layeredGeometry,
  runElk,
  type LayoutMode,
} from "../layout/shared";
import type { WorkflowEdgeView, WorkflowNodeKind, WorkflowNodeView } from "../types";

/** Same physical card as the task DAG; workflows widen and pan, they never shrink. */
export const WF_NODE_W = CARD_W;
export const WF_NODE_H = CARD_H;
const LAYER_GAP = 64;
const ROW_GAP = 24;

export interface WorkflowLayoutEdge extends WorkflowEdgeView {
  id: string;
  /** Points backwards (or at itself) in declaration order: a retry / repair loop. */
  back: boolean;
}

export interface WorkflowLayout {
  signature: string;
  mode: LayoutMode;
  positions: Map<string, Point>;
  heights: Map<string, number>;
  routes: Map<string, Point[]>;
  width: number;
  height: number;
}

/** `human_gate` / `gate_*` → gate; `validate_*` → validator; every LLM or execute node → stage. */
export function classifyNode(name: string, kind?: string): WorkflowNodeKind {
  const raw = String(kind || "").toLowerCase();
  const id = String(name || "").toLowerCase();
  if (raw === "human_gate" || raw.startsWith("gate") || id.startsWith("gate_")) return "gate";
  if (raw.startsWith("validate") || id.startsWith("validate_")) return "validator";
  return "stage";
}

/** Dedupe nodes/edges, drop dangling edges, and flag back-edges by declaration order. */
export function normalizeWorkflow(
  rawNodes: WorkflowNodeView[],
  rawEdges: WorkflowEdgeView[],
): { nodes: WorkflowNodeView[]; edges: WorkflowLayoutEdge[] } {
  const order = new Map<string, number>();
  const nodes = (rawNodes || []).filter((node) => {
    if (!node || typeof node.id !== "string" || !node.id || order.has(node.id)) return false;
    order.set(node.id, order.size);
    return true;
  });
  const seen = new Set<string>();
  const edges: WorkflowLayoutEdge[] = [];
  (rawEdges || []).forEach((edge) => {
    if (!edge || !order.has(edge.source) || !order.has(edge.target)) return;
    const id = edgeKey(edge.source, edge.target);
    if (seen.has(id)) return;
    seen.add(id);
    const back = (order.get(edge.target) as number) <= (order.get(edge.source) as number);
    edges.push({ ...edge, id, back });
  });
  return { nodes, edges };
}

/** Layout reruns only when this changes; status, trajectory and selection patch in place. */
export function workflowSignature(
  runId: string,
  nodes: WorkflowNodeView[],
  edges: WorkflowLayoutEdge[],
): string {
  return JSON.stringify([
    runId || "",
    nodes.map((node) => node.id),
    edges.map((edge) => edge.id).sort(),
  ]);
}

/** Longest-path layers over forward edges only, so retry loops can never recurse forever. */
export function workflowLayers(nodes: WorkflowNodeView[], edges: WorkflowLayoutEdge[]): Map<string, number> {
  const layers = new Map(nodes.map((node) => [node.id, 0]));
  nodes.forEach((node) => {
    const base = layers.get(node.id) as number;
    edges.forEach((edge) => {
      if (edge.back || edge.source !== node.id) return;
      layers.set(edge.target, Math.max(layers.get(edge.target) as number, base + 1));
    });
  });
  return layers;
}

/** Deterministic, cycle-safe layered geometry; used when ELK is unavailable or fails. */
export function workflowFallback(
  nodes: WorkflowNodeView[],
  edges: WorkflowLayoutEdge[],
  signature: string,
): WorkflowLayout {
  const layers = workflowLayers(nodes, edges);
  const geometry = layeredGeometry(
    nodes.map((node) => ({ id: node.id, layer: layers.get(node.id) ?? 0 })),
    edges,
    { layerGap: LAYER_GAP, rowGap: ROW_GAP, top: 0 },
  );
  return { signature, mode: "fallback", ...geometry };
}

const ROOT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.cycleBreaking.strategy": "MODEL_ORDER",
  "elk.layered.layering.strategy": "LONGEST_PATH_SOURCE",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  // Keep the stage chain on one editorial baseline (seeded by layer/row hints below);
  // skip and back edges take their own orthogonal channels above and below it.
  "elk.layered.nodePlacement.strategy": "INTERACTIVE",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.mergeEdges": "false",
  "elk.layered.spacing.nodeNodeBetweenLayers": String(LAYER_GAP),
  "elk.layered.spacing.edgeNodeBetweenLayers": "18",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "12",
  "elk.spacing.nodeNode": String(ROW_GAP),
  "elk.spacing.edgeNode": "16",
  "elk.spacing.edgeEdge": "12",
  "elk.spacing.portPort": "14",
  "elk.padding": "[top=0,left=0,bottom=0,right=0]",
  "elk.randomSeed": "1",
};

/** ELK Layered in declaration order with distributed west-in / east-out ports per edge. */
export async function elkWorkflowLayout(
  nodes: WorkflowNodeView[],
  edges: WorkflowLayoutEdge[],
  signature: string,
): Promise<WorkflowLayout> {
  const layers = workflowLayers(nodes, edges);
  const heights = cardHeights(nodes, edges);
  const cursorY = new Map<number, number>();
  const hint = new Map(
    nodes.map((node) => {
      const layer = layers.get(node.id) ?? 0;
      const y = cursorY.get(layer) || 0;
      const height = heights.get(node.id) ?? WF_NODE_H;
      cursorY.set(layer, y + height + ROW_GAP);
      return [node.id, { x: layer * (WF_NODE_W + LAYER_GAP), y, height }];
    }),
  );
  // Feed ELK edges in declaration order (source, then target) so crossing minimisation keeps
  // the shortest hop on the baseline and stacks longer skips outward, independent of host order.
  const order = new Map(nodes.map((node, index) => [node.id, index]));
  const rank = (edge: WorkflowLayoutEdge): number =>
    (order.get(edge.source) as number) * nodes.length + (order.get(edge.target) as number);
  const sorted = [...edges].sort((a, b) => rank(a) - rank(b));
  const ports = new Map<string, ElkPort[]>(nodes.map((node) => [node.id, []]));
  sorted.forEach((edge) => {
    ports.get(edge.source)?.push({
      id: `${edge.id}#out`, width: 0, height: 0, layoutOptions: { "elk.port.side": "EAST" },
    });
    ports.get(edge.target)?.push({
      id: `${edge.id}#in`, width: 0, height: 0, layoutOptions: { "elk.port.side": "WEST" },
    });
  });
  const graph: ElkNode = {
    id: "ariadne-workflow",
    layoutOptions: ROOT_OPTIONS,
    children: nodes.map((node) => ({
      id: node.id,
      width: WF_NODE_W,
      ...(hint.get(node.id) || { x: 0, y: 0, height: WF_NODE_H }),
      ports: ports.get(node.id),
      layoutOptions: { "elk.portConstraints": "FIXED_SIDE", "elk.spacing.portPort": "10" },
    })),
    edges: sorted.map(
      (edge): ElkExtendedEdge => ({ id: edge.id, sources: [`${edge.id}#out`], targets: [`${edge.id}#in`] }),
    ),
  };
  const geometry = await runElk(graph, nodes.length, edges.length, 0, heights);
  return { signature, mode: "elk", ...geometry };
}

/** Never rejects: any ELK failure degrades to the deterministic layered fallback. */
export async function computeWorkflowLayout(
  nodes: WorkflowNodeView[],
  edges: WorkflowLayoutEdge[],
  signature: string,
): Promise<WorkflowLayout> {
  try {
    return await elkWorkflowLayout(nodes, edges, signature);
  } catch (error) {
    console.warn("[AriadneGraph] ELK workflow layout failed; using layered fallback", error);
    return workflowFallback(nodes, edges, signature);
  }
}
