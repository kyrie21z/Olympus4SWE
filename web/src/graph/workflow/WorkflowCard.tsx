import { memo, type FocusEvent, type KeyboardEvent } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import type { NodeLineage } from "../lineage";
import type { WorkflowNodeKind, WorkflowNodeView } from "../types";
import { WF_NODE_H, WF_NODE_W } from "./layout";

export interface WorkflowCardData extends Record<string, unknown> {
  view: WorkflowNodeView;
  height: number;
  kind: WorkflowNodeKind;
  selected: boolean;
  lineage?: NodeLineage;
  select: (id: string) => void;
  hover: (id: string | null) => void;
  focus: (id: string | null) => void;
}

export type WorkflowCardNode = Node<WorkflowCardData, "workflow">;

const KIND_BADGE: Record<WorkflowNodeKind, string> = { gate: "关口", validator: "校验", stage: "" };

/** Status → shared card tone; `skipped` gets its own recessed, dashed treatment. */
export function workflowTone(status: string): string {
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

/** Editorial three-row workflow card: title + kind badge | node name | status capsule. */
function WorkflowCardComponent({ data }: NodeProps<WorkflowCardNode>) {
  const { view, height, kind, selected, lineage, select, hover, focus } = data;
  const badge = KIND_BADGE[kind];
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    select(view.id);
  };
  const onFocus = (event: FocusEvent<HTMLDivElement>) => {
    if (event.currentTarget.matches(":focus-visible")) focus(view.id);
  };
  return (
    <div
      className="ag-card ag-wf-card nopan"
      style={{ width: WF_NODE_W, height: height || WF_NODE_H }}
      data-workflow-node={view.id}
      data-status={view.status}
      data-tone={workflowTone(view.status)}
      data-node-kind={kind}
      data-lineage={lineage || undefined}
      role="button"
      tabIndex={0}
      aria-selected={selected ? "true" : "false"}
      aria-label={badge ? `${view.ariaLabel} · ${badge}` : view.ariaLabel}
      title={view.title || undefined}
      onClick={() => select(view.id)}
      onKeyDown={onKeyDown}
      onMouseEnter={() => hover(view.id)}
      onMouseLeave={() => hover(null)}
      onFocus={onFocus}
      onBlur={() => focus(null)}
    >
      <Handle className="ag-handle" type="target" position={Position.Left} isConnectable={false} />
      <div className="ag-card-head">
        <span className="ag-card-title">{view.label}</span>
        {badge ? (
          <span className="ag-card-kind" data-kind={kind}>
            {badge}
          </span>
        ) : null}
      </div>
      <span className="ag-card-meta">{view.id}</span>
      <span className="ag-card-status">{view.statusText}</span>
      <Handle className="ag-handle" type="source" position={Position.Right} isConnectable={false} />
    </div>
  );
}

export const WorkflowCard = memo(WorkflowCardComponent);
