import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { dispatchToHost } from "../island";
import { computeLineage, type LineageGraph } from "../lineage";

export interface AsyncGraphState<L> {
  current: L | null;
  lineage: LineageGraph | null;
  select: (id: string) => void;
  hover: (id: string | null) => void;
  focus: (id: string | null) => void;
}

/**
 * Shared lifecycle hook for both Topology graphs: version-guarded async layout,
 * host selection dispatch, and decoupled pointer-hover + keyboard-focus lineage state.
 */
export function useGraphIsland<G extends { nodes: readonly unknown[]; edges: readonly { source: string; target: string }[] }, L extends { signature: string }>(
  graph: G,
  signature: string,
  compute: (graph: G, signature: string) => Promise<L>,
  onSelect?: (id: string) => void,
): AsyncGraphState<L> {
  const latestGraph = useRef(graph);
  latestGraph.current = graph;
  const computeRef = useRef(compute);
  computeRef.current = compute;
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  const [layout, setLayout] = useState<L | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const version = useRef(0);

  useEffect(() => {
    setHoveredId(null);
    setFocusedId(null);
    const ticket = ++version.current;
    const snapshot = latestGraph.current;
    if (!snapshot.nodes.length) return undefined;
    let cancelled = false;
    void computeRef.current(snapshot, signature).then((result) => {
      if (!cancelled && ticket === version.current) setLayout(result);
    });
    return () => {
      cancelled = true;
    };
  }, [signature]);

  const select = useCallback((id: string) => {
    const handler = onSelectRef.current;
    if (handler) dispatchToHost(() => handler(id));
  }, []);

  const hover = useCallback((id: string | null) => {
    setHoveredId(id);
  }, []);

  const focus = useCallback((id: string | null) => {
    setFocusedId(id);
  }, []);

  const current = layout && layout.signature === signature ? layout : null;
  const activeId = hoveredId ?? focusedId;
  const lineage = useMemo(() => computeLineage(graph.edges, activeId), [graph.edges, activeId]);

  return { current, lineage, select, hover, focus };
}
