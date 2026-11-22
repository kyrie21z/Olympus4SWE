import type { ElkExtendedEdge, ElkNode, ElkPort } from "elkjs/lib/elk-api";

import type { Point } from "../geometry";
import {
  CARD_H,
  CARD_W,
  cardHeights,
  columnsOf,
  edgeKey,
  hasCycle,
  layeredGeometry,
  runElk,
  type LayoutMode,
} from "../layout/shared";
import type { DagEdgeView, DagNodeView } from "../types";

export { edgeKey, hasCycle };

/** Fixed 1:1 physical card size, shared with the workflow graph. */
export const NODE_W = CARD_W;
export const NODE_H = CARD_H;
/** Vertical band above the first card row that hosts the Wave column headers. */
export const WAVE_BAND = 36;
const LAYER_GAP = 96;
const ROW_GAP = 24;

export interface LayoutEdge extends DagEdgeView {
  id: string;
}

export interface DagLayout {
  signature: string;
  mode: LayoutMode;
  positions: Map<string, Point>;
  heights: Map<string, number>;
  routes: Map<string, Point[]>;
  waves: { layer: number; x: number }[];
  width: number;
  height: number;
}

/** Cycle-safe longest-path wave for nodes whose host layer is missing or invalid. */
function fillLayers(nodes: DagNodeView[], edges: LayoutEdge[]): Map<string, number> {
  const parents = new Map<string, string[]>();
  edges.forEach((edge) => parents.set(edge.target, [...(parents.get(edge.target) || []), edge.source]));
  const memo = new Map<string, number>();
  const given = new Map(nodes.map((node) => [node.id, node.layer]));
  const visit = (id: string, visiting: Set<string>): number => {
    const host = given.get(id);
    if (typeof host === "number" && Number.isInteger(host) && host >= 0) return host;
    if (memo.has(id)) return memo.get(id) as number;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const deps = (parents.get(id) || []).map((parent) => visit(parent, visiting));
    visiting.delete(id);
    const layer = deps.length ? Math.max(...deps) + 1 : 0;
    memo.set(id, layer);
    return layer;
  };
  return new Map(nodes.map((node) => [node.id, visit(node.id, new Set())]));
}

/** Dedupe nodes/edges, drop dangling edges, and give every node an integer wave. */
export function normalizeGraph(
  rawNodes: DagNodeView[],
  rawEdges: DagEdgeView[],
): { nodes: DagNodeView[]; edges: LayoutEdge[] } {
  const seen = new Set<string>();
  const unique = (rawNodes || []).filter((node) => {
    if (!node || typeof node.id !== "string" || seen.has(node.id)) return false;
    seen.add(node.id);
    return true;
  });
  const edgeIds = new Set<string>();
  const edges: LayoutEdge[] = [];
  (rawEdges || []).forEach((edge) => {
    if (!edge || !seen.has(edge.source) || !seen.has(edge.target)) return;
    const id = edgeKey(edge.source, edge.target);
    if (edgeIds.has(id)) return;
    edgeIds.add(id);
    edges.push({ ...edge, id });
  });
  const layers = fillLayers(unique, edges);
  const nodes = unique.map((node) => ({ ...node, layer: layers.get(node.id) ?? 0 }));
  return { nodes, edges };
}

/** Layout reruns only when this changes; status/selection patches keep the viewport. */
export function topologySignature(runId: string, nodes: DagNodeView[], edges: LayoutEdge[]): string {
  const nodePart = nodes.map((node) => `${node.id}@${node.layer}`).sort();
  const edgePart = edges.map((edge) => edge.id).sort();
  return JSON.stringify([runId || "", nodePart, edgePart]);
}

function wavesFrom(nodes: DagNodeView[], positions: Map<string, Point>): DagLayout["waves"] {
  const byLayer = new Map<number, number>();
  nodes.forEach((node) => {
    const pos = positions.get(node.id);
    if (pos && !byLayer.has(node.layer)) byLayer.set(node.layer, pos.x);
  });
  return Array.from(byLayer, ([layer, x]) => ({ layer, x })).sort((a, b) => a.layer - b.layer);
}

/** Deterministic layered coordinates + simple orthogonal routes; never throws, tolerates cycles. */
export function layeredFallback(
  nodes: DagNodeView[],
  edges: LayoutEdge[],
  signature: string,
): DagLayout {
  const geometry = layeredGeometry(nodes, edges, { layerGap: LAYER_GAP, rowGap: ROW_GAP, top: WAVE_BAND });
  return { signature, mode: "fallback", ...geometry, waves: wavesFrom(nodes, geometry.positions) };
}

const ROOT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.layering.strategy": "INTERACTIVE",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.mergeEdges": "false",
  "elk.layered.spacing.nodeNodeBetweenLayers": String(LAYER_GAP),
  "elk.layered.spacing.edgeNodeBetweenLayers": "20",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "12",
  "elk.spacing.nodeNode": String(ROW_GAP),
  "elk.spacing.edgeNode": "18",
  "elk.spacing.edgeEdge": "12",
  "elk.spacing.portPort": "14",
  "elk.padding": "[top=0,left=0,bottom=0,right=0]",
  "elk.randomSeed": "1",
};

/** ELK Layered with waves pinned to `node.layer` and distributed west-in / east-out ports. */
export async function elkLayout(
  nodes: DagNodeView[],
  edges: LayoutEdge[],
  signature: string,
): Promise<DagLayout> {
  const column = columnsOf(nodes);
  const heights = cardHeights(nodes, edges);
  const cursorY = new Map<number, number>();
  const ports = new Map<string, ElkPort[]>(nodes.map((node) => [node.id, []]));
  edges.forEach((edge) => {
    ports.get(edge.source)?.push({
      id: `${edge.id}#out`, width: 0, height: 0, layoutOptions: { "elk.port.side": "EAST" },
    });
    ports.get(edge.target)?.push({
      id: `${edge.id}#in`, width: 0, height: 0, layoutOptions: { "elk.port.side": "WEST" },
    });
  });
  const graph: ElkNode = {
    id: "ariadne-dag",
    layoutOptions: ROOT_OPTIONS,
    children: nodes.map((node) => {
      const col = column.get(node.layer) ?? 0;
      const y = cursorY.get(col) || 0;
      const height = heights.get(node.id) ?? NODE_H;
      cursorY.set(col, y + height + ROW_GAP);
      return {
        id: node.id,
        width: NODE_W,
        height,
        x: col * (NODE_W + LAYER_GAP),
        y,
        ports: ports.get(node.id),
        layoutOptions: {
          "elk.portConstraints": "FIXED_SIDE",
          "elk.spacing.portPort": "10",
        },
      };
    }),
    edges: edges.map(
      (edge): ElkExtendedEdge => ({ id: edge.id, sources: [`${edge.id}#out`], targets: [`${edge.id}#in`] }),
    ),
  };
  const geometry = await runElk(graph, nodes.length, edges.length, WAVE_BAND, heights);
  const layout: DagLayout = {
    signature,
    mode: "elk",
    ...geometry,
    waves: wavesFrom(nodes, geometry.positions),
  };
  assertWavesAligned(nodes, layout);
  return layout;
}

/** ELK must honour the host's waves: one shared x per wave, columns ordered by wave. */
function assertWavesAligned(nodes: DagNodeView[], layout: DagLayout): void {
  const xs = new Map<number, number>();
  nodes.forEach((node) => {
    const x = (layout.positions.get(node.id) as Point).x;
    const first = xs.get(node.layer);
    if (first === undefined) xs.set(node.layer, x);
    else if (Math.abs(first - x) > 0.5) throw new Error(`wave ${node.layer} is not column-aligned`);
  });
  const ordered = Array.from(xs).sort((a, b) => a[0] - b[0]);
  ordered.forEach(([layer, x], index) => {
    if (index && x <= ordered[index - 1][1]) throw new Error(`wave ${layer} is out of order`);
  });
}

/** Never rejects: cyclic input or any ELK failure degrades to the deterministic fallback. */
export async function computeDagLayout(
  nodes: DagNodeView[],
  edges: LayoutEdge[],
  signature: string,
): Promise<DagLayout> {
  if (hasCycle(nodes, edges)) return layeredFallback(nodes, edges, signature);
  try {
    return await elkLayout(nodes, edges, signature);
  } catch (error) {
    console.warn("[AriadneGraph] ELK layout failed; using layered fallback", error);
    return layeredFallback(nodes, edges, signature);
  }
}
