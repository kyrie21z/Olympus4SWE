import { useCallback, useMemo } from "react";
import { Position, ViewportPortal, type EdgeTypes, type NodeHandle, type NodeTypes } from "@xyflow/react";

import { GraphCanvas, frameHeight, useMarkerPrefix } from "../canvas/GraphCanvas";
import { useGraphIsland } from "../canvas/useGraphIsland";
import { OrthogonalEdge, edgeTone, type OrthogonalEdgeType } from "../edges/OrthogonalEdge";
import type { IslandController } from "../island";
import { edgeLineage, nodeLineage } from "../lineage";
import type { DagGraphProps } from "../types";
import {
  NODE_H,
  NODE_W,
  computeDagLayout,
  layeredFallback,
  normalizeGraph,
  topologySignature,
  type DagLayout,
} from "./layout";
import { TaskCard, type TaskCardNode } from "./TaskCard";

const NODE_TYPES: NodeTypes = { task: TaskCard };
const EDGE_TYPES: EdgeTypes = { orthogonal: OrthogonalEdge };

/** Adaptive frame bounds (px). */
const MIN_HEIGHT = 300;
const MAX_HEIGHT = 540;

const handlesFor = (height: number): NodeHandle[] => [
  { type: "target", position: Position.Left, x: 0, y: height / 2, width: 1, height: 1 },
  { type: "source", position: Position.Right, x: NODE_W - 1, y: height / 2, width: 1, height: 1 },
];

export function DagGraph(props: DagGraphProps & { controller: IslandController }) {
  const { runId, selectedId, onSelect, controller } = props;
  const markerPrefix = useMarkerPrefix();

  const graph = useMemo(() => normalizeGraph(props.nodes, props.edges), [props.nodes, props.edges]);
  const signature = useMemo(
    () => topologySignature(runId, graph.nodes, graph.edges),
    [runId, graph],
  );
  const runLayout = useCallback(
    (snapshot: typeof graph, sig: string): Promise<DagLayout> =>
      computeDagLayout(snapshot.nodes, snapshot.edges, sig),
    [],
  );
  const { current, lineage, select, hover, focus } = useGraphIsland(
    graph,
    signature,
    runLayout,
    onSelect,
  );

  const nodes = useMemo<TaskCardNode[]>(() => {
    if (!current) return [];
    return graph.nodes.map((view) => {
      const height = current.heights.get(view.id) ?? NODE_H;
      return {
        id: view.id,
        type: "task" as const,
        position: current.positions.get(view.id) || { x: 0, y: 0 },
        width: NODE_W,
        height,
        handles: handlesFor(height),
        draggable: false,
        connectable: false,
        selectable: false,
        focusable: false,
        data: {
          view,
          height,
          selected: view.id === selectedId,
          lineage: nodeLineage(lineage, view.id),
          select,
          hover,
          focus,
        },
      };
    });
  }, [current, graph, lineage, selectedId, select, hover, focus]);

  const edges = useMemo<OrthogonalEdgeType[]>(() => {
    if (!current) return [];
    return graph.edges.map((edge) => {
      const tone = edgeTone(edge);
      const relation = edgeLineage(lineage, edge.source, edge.target);
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "orthogonal" as const,
        selectable: false,
        focusable: false,
        zIndex: relation === "related" ? 2 : tone === "default" ? 0 : 1,
        data: {
          points: current.routes.get(edge.id) || [],
          tone,
          markerId: `${markerPrefix}-${tone}`,
          lineage: relation,
        },
      };
    });
  }, [current, graph, lineage, markerPrefix]);

  const contentHeight = useMemo(() => {
    if (current) return current.height;
    return graph.nodes.length ? layeredFallback(graph.nodes, graph.edges, signature).height : 0;
  }, [current, graph, signature]);

  if (!graph.nodes.length) {
    return (
      <div className="ag-island" data-graph="dag" data-empty="true">
        <p className="ag-notice" role="status">
          {props.emptyMessage || "当前运行尚未生成任务 DAG"}
        </p>
      </div>
    );
  }

  return (
    <GraphCanvas
      graph="dag"
      layout={current}
      height={frameHeight(contentHeight, MIN_HEIGHT, MAX_HEIGHT)}
      ariaLabel={props.ariaLabel || "任务依赖有向无环图"}
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      edgeTypes={EDGE_TYPES}
      markerPrefix={markerPrefix}
      controller={controller}
    >
      {current ? (
        <ViewportPortal>
          <div className="ag-waves" aria-hidden="true">
            {current.waves.map((wave) => (
              <div
                key={wave.layer}
                className="ag-wave"
                data-wave={wave.layer}
                style={{ width: NODE_W, transform: `translate(${wave.x}px, 0px)` }}
              >
                Wave {wave.layer}
              </div>
            ))}
          </div>
        </ViewportPortal>
      ) : null}
    </GraphCanvas>
  );
}
