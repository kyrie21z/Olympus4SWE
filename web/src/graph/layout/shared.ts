import ELK from "elkjs/lib/elk.bundled.js";
import type { ElkNode } from "elkjs/lib/elk-api";

import type { Point } from "../geometry";

/**
 * One physical card size for every Ariadne graph (task DAG and workflow alike), so both
 * Topology graphs read at the same 1:1 typography. CSS reads the same numbers via inline style.
 */
export const CARD_W = 184;
export const CARD_H = 76;
/** Minimum vertical port step (px) so fixed 7.2px arrowheads never overlap on high fan-in cards. */
const PORT_STEP = 12;

export type LayoutMode = "elk" | "fallback";

export interface LayoutEdgeRef {
  id: string;
  source: string;
  target: string;
}

export interface LayerNodeRef {
  id: string;
  layer: number;
}

/** Card positions, per-node heights, and edge polylines in graph coordinates. */
export interface Geometry {
  positions: Map<string, Point>;
  heights: Map<string, number>;
  routes: Map<string, Point[]>;
  width: number;
  height: number;
}

export interface LayerBox {
  layerGap: number;
  rowGap: number;
  /** Vertical offset of the first card row (e.g. a header band). */
  top: number;
}

export const edgeKey = (source: string, target: string): string => JSON.stringify([source, target]);

/** Compute card heights: standard 76px for normal fan-in/out, taller only when side ports exceed capacity. */
export function cardHeights(
  nodes: readonly { id: string }[],
  edges: readonly LayoutEdgeRef[],
): Map<string, number> {
  const incoming = new Map<string, number>();
  const outgoing = new Map<string, number>();
  edges.forEach((edge) => {
    outgoing.set(edge.source, (outgoing.get(edge.source) || 0) + 1);
    incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1);
  });
  return new Map(
    nodes.map((node) => {
      const maxSide = Math.max(incoming.get(node.id) || 0, outgoing.get(node.id) || 0);
      return [node.id, Math.max(CARD_H, (maxSide + 1) * PORT_STEP)];
    }),
  );
}

/** Kahn's algorithm: true when the edges contain a directed cycle. */
export function hasCycle(nodes: { id: string }[], edges: LayoutEdgeRef[]): boolean {
  const indegree = new Map(nodes.map((node) => [node.id, 0]));
  const children = new Map<string, string[]>();
  edges.forEach((edge) => {
    indegree.set(edge.target, (indegree.get(edge.target) || 0) + 1);
    children.set(edge.source, [...(children.get(edge.source) || []), edge.target]);
  });
  const queue = nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id);
  let visited = 0;
  while (queue.length) {
    const id = queue.shift() as string;
    visited += 1;
    (children.get(id) || []).forEach((child) => {
      const left = (indegree.get(child) || 0) - 1;
      indegree.set(child, left);
      if (left === 0) queue.push(child);
    });
  }
  return visited !== nodes.length;
}

/** Map sparse layer numbers onto contiguous column indices. */
export function columnsOf(nodes: LayerNodeRef[]): Map<number, number> {
  const layers = Array.from(new Set(nodes.map((node) => node.layer))).sort((a, b) => a - b);
  return new Map(layers.map((layer, index) => [layer, index]));
}

const RAIL_STEP = 10;
const RAIL_GAP = 16;

/**
 * Deterministic layered coordinates + simple orthogonal routes; never throws, tolerates cycles.
 * Adjacent-column edges bend once in the gap between the columns; skip-column, same-column and
 * backward edges drop to their own rail below every card so no path crosses a card.
 */
export function layeredGeometry(
  nodes: LayerNodeRef[],
  edges: LayoutEdgeRef[],
  box: LayerBox,
): Geometry {
  const column = columnsOf(nodes);
  const heights = cardHeights(nodes, edges);
  const cursorY = new Map<number, number>();
  const positions = new Map<string, Point>();
  const colOf = new Map<string, number>();
  nodes.forEach((node) => {
    const col = column.get(node.layer) ?? 0;
    const y = cursorY.get(col) ?? box.top;
    const h = heights.get(node.id) ?? CARD_H;
    cursorY.set(col, y + h + box.rowGap);
    colOf.set(node.id, col);
    positions.set(node.id, { x: col * (CARD_W + box.layerGap), y });
  });
  const portY = (ids: string[], id: string, top: number, h: number): number =>
    top + (h * (ids.indexOf(id) + 1)) / (ids.length + 1);
  const byY = (key: "source" | "target") => (a: LayoutEdgeRef, b: LayoutEdgeRef) =>
    (positions.get(a[key]) as Point).y - (positions.get(b[key]) as Point).y || (a.id < b.id ? -1 : 1);
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  nodes.forEach((node) => {
    incoming.set(node.id, edges.filter((e) => e.target === node.id).sort(byY("source")).map((e) => e.id));
    outgoing.set(node.id, edges.filter((e) => e.source === node.id).sort(byY("target")).map((e) => e.id));
  });
  let bottom = Math.max(
    ...nodes.map((node) => ((positions.get(node.id) as Point).y + (heights.get(node.id) ?? CARD_H))),
  );
  const routes = new Map<string, Point[]>();
  let rail = 0;
  edges.forEach((edge, index) => {
    const s = positions.get(edge.source) as Point;
    const t = positions.get(edge.target) as Point;
    const sy = portY(outgoing.get(edge.source) || [], edge.id, s.y, heights.get(edge.source) ?? CARD_H);
    const ty = portY(incoming.get(edge.target) || [], edge.id, t.y, heights.get(edge.target) ?? CARD_H);
    const start = { x: s.x + CARD_W, y: sy };
    const end = { x: t.x, y: ty };
    if ((colOf.get(edge.target) ?? 0) - (colOf.get(edge.source) ?? 0) === 1) {
      const channel = t.x - box.layerGap / 2 + ((index % 5) - 2) * 6;
      routes.set(edge.id, [start, { x: channel, y: sy }, { x: channel, y: ty }, end]);
      return;
    }
    rail += 1;
    const under = bottom + RAIL_GAP + rail * RAIL_STEP;
    const out = start.x + RAIL_GAP + (rail % 3) * 4;
    const back = end.x - RAIL_GAP - (rail % 3) * 4;
    routes.set(edge.id, [
      start,
      { x: out, y: sy },
      { x: out, y: under },
      { x: back, y: under },
      { x: back, y: ty },
      end,
    ]);
  });
  if (rail) bottom += RAIL_GAP + rail * RAIL_STEP + 8;
  const width = Math.max(...Array.from(positions.values(), (pos) => pos.x + CARD_W));
  return { positions, heights, routes, width, height: bottom };
}

let engine: InstanceType<typeof ELK> | null = null;

/**
 * Run ELK and read back card positions plus the exact orthogonal edge sections, normalised to
 * start at (0, top). Throws when ELK drops any node or edge so callers can fall back.
 */
export async function runElk(
  graph: ElkNode,
  nodeCount: number,
  edgeCount: number,
  top: number,
  heights: Map<string, number>,
): Promise<Geometry> {
  engine = engine || new ELK();
  const result = await engine.layout(graph);
  const positions = new Map<string, Point>();
  (result.children || []).forEach((child) => {
    positions.set(child.id, { x: child.x ?? 0, y: child.y ?? 0 });
  });
  const routes = new Map<string, Point[]>();
  (result.edges || []).forEach((edge) => {
    const sections = edge.sections || [];
    if (!sections.length) return;
    const points: Point[] = [];
    sections.forEach((section) => {
      points.push(section.startPoint, ...(section.bendPoints || []), section.endPoint);
    });
    routes.set(edge.id, points.map((point) => ({ x: point.x, y: point.y })));
  });
  if (positions.size !== nodeCount || routes.size !== edgeCount) {
    throw new Error("ELK returned an incomplete layout");
  }
  const all = [
    ...Array.from(positions.entries()).flatMap(([id, p]) => [
      p,
      { x: p.x + CARD_W, y: p.y + (heights.get(id) ?? CARD_H) },
    ]),
    ...Array.from(routes.values()).flat(),
  ];
  const minX = Math.min(...all.map((p) => p.x));
  const minY = Math.min(...all.map((p) => p.y));
  const shift = (p: Point): Point => ({ x: p.x - minX, y: p.y - minY + top });
  positions.forEach((pos, id) => positions.set(id, shift(pos)));
  routes.forEach((points, id) => routes.set(id, points.map(shift)));
  return {
    positions,
    heights,
    routes,
    width: Math.max(...all.map((p) => p.x)) - minX,
    height: Math.max(...all.map((p) => p.y)) - minY + top,
  };
}
