import "@xyflow/react/dist/base.css";
import "../styles/graph.css";

import { useRef } from "react";
import { ReactFlowProvider } from "@xyflow/react";

import { DagGraph } from "./dag/DagGraph";
import { registerGraphHarness } from "./harness";
import type { IslandController } from "./island";
import type { DagGraphProps, WorkflowGraphProps } from "./types";
import { WorkflowGraph } from "./workflow/WorkflowGraph";

export type * from "./types";

function useController(): IslandController {
  return useRef<IslandController>({ fit: null }).current;
}

export function DagGraphView(props: DagGraphProps) {
  const controller = useController();
  return (
    <ReactFlowProvider>
      <DagGraph {...props} controller={controller} />
    </ReactFlowProvider>
  );
}

export function WorkflowGraphView(props: WorkflowGraphProps) {
  const controller = useController();
  return (
    <ReactFlowProvider>
      <WorkflowGraph {...props} controller={controller} />
    </ReactFlowProvider>
  );
}

registerGraphHarness();
