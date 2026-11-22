import { memo, type FocusEvent, type KeyboardEvent } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import type { NodeLineage } from "../lineage";
import type { DagNodeView } from "../types";
import { NODE_H, NODE_W } from "./layout";

export interface TaskCardData extends Record<string, unknown> {
  view: DagNodeView;
  height: number;
  selected: boolean;
  lineage?: NodeLineage;
  select: (id: string) => void;
  hover: (id: string | null) => void;
  focus: (id: string | null) => void;
}

export type TaskCardNode = Node<TaskCardData, "task">;

/** Editorial three-row task card: id + pill | role · kind | status capsule. */
function TaskCardComponent({ data }: NodeProps<TaskCardNode>) {
  const { view, height, selected, lineage, select, hover, focus } = data;
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
      className="ag-card ag-task-card nopan"
      style={{ width: NODE_W, height: height || NODE_H }}
      data-task-id={view.id}
      data-status={view.status}
      data-tone={view.tone}
      data-issue={view.issue || ""}
      data-risk={view.risk ? "true" : "false"}
      data-lineage={lineage || undefined}
      role="button"
      tabIndex={0}
      aria-selected={selected ? "true" : "false"}
      aria-label={view.ariaLabel}
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
        <span className="ag-card-id">{view.id}</span>
        {view.pill ? (
          <span className="ag-card-pill" data-tone={view.pill.tone || "muted"}>
            {view.pill.text}
          </span>
        ) : null}
      </div>
      <span className="ag-card-meta">{view.meta}</span>
      <span className="ag-card-status">{view.statusText}</span>
      <Handle className="ag-handle" type="source" position={Position.Right} isConnectable={false} />
    </div>
  );
}

export const TaskCard = memo(TaskCardComponent);
