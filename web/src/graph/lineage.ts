export type NodeLineage = "focus" | "related" | "dim";
export type EdgeLineage = "related" | "dim";

export interface LineageGraph {
  focusId: string;
  ancestors: Set<string>;
  descendants: Set<string>;
}

function walk(start: string, adjacency: Map<string, string[]>): Set<string> {
  const visited = new Set<string>([start]);
  const queue = [start];
  while (queue.length) {
    const current = queue.shift() as string;
    for (const next of adjacency.get(current) || []) {
      if (visited.has(next)) continue;
      visited.add(next);
      queue.push(next);
    }
  }
  return visited;
}

/** Cycle-safe upstream + downstream cone for the hovered or keyboard-focused node. */
export function computeLineage(
  edges: ReadonlyArray<{ source: string; target: string }>,
  focusId: string | null,
): LineageGraph | null {
  if (!focusId) return null;
  const parents = new Map<string, string[]>();
  const children = new Map<string, string[]>();
  for (const edge of edges) {
    const prev = parents.get(edge.target);
    if (prev) prev.push(edge.source);
    else parents.set(edge.target, [edge.source]);

    const next = children.get(edge.source);
    if (next) next.push(edge.target);
    else children.set(edge.source, [edge.target]);
  }
  return {
    focusId,
    ancestors: walk(focusId, parents),
    descendants: walk(focusId, children),
  };
}

export function nodeLineage(lineage: LineageGraph | null, id: string): NodeLineage | undefined {
  if (!lineage) return undefined;
  if (id === lineage.focusId) return "focus";
  if (lineage.ancestors.has(id) || lineage.descendants.has(id)) return "related";
  return "dim";
}

export function edgeLineage(
  lineage: LineageGraph | null,
  source: string,
  target: string,
): EdgeLineage | undefined {
  if (!lineage) return undefined;
  const onUpstream = lineage.ancestors.has(source) && lineage.ancestors.has(target);
  const onDownstream = lineage.descendants.has(source) && lineage.descendants.has(target);
  return onUpstream || onDownstream ? "related" : "dim";
}
