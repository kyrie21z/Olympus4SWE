import { memo } from "react";
import type { Edge, EdgeProps } from "@xyflow/react";

import { roundedPolylinePath, type Point } from "../geometry";
import type { EdgeLineage } from "../lineage";
import type { EdgeTone } from "../types";

export interface OrthogonalEdgeData extends Record<string, unknown> {
  points: Point[];
  tone: EdgeTone;
  markerId: string;
  lineage?: EdgeLineage;
  /** Extra stable hooks rendered as `data-<key>` on the edge group (e.g. `traversed`). */
  attrs?: Record<string, string>;
}

export type OrthogonalEdgeType = Edge<OrthogonalEdgeData, "orthogonal">;

const dataAttrs = (attrs: Record<string, string> | undefined): Record<string, string> =>
  Object.fromEntries(Object.entries(attrs || {}).map(([key, value]) => [`data-${key}`, value]));

/** Renders the exact layout polyline (ELK `sections`) with rounded corners and a tone-matched head. */
function OrthogonalEdgeComponent({ source, target, data }: EdgeProps<OrthogonalEdgeType>) {
  if (!data || data.points.length < 2) return null;
  return (
    <g
      className="ag-edge"
      data-tone={data.tone}
      data-edge-source={source}
      data-edge-target={target}
      data-lineage={data.lineage || undefined}
      {...dataAttrs(data.attrs)}
    >
      <path
        className="ag-edge-path"
        d={roundedPolylinePath(data.points, 6)}
        markerEnd={`url(#${data.markerId})`}
      />
    </g>
  );
}

export const OrthogonalEdge = memo(OrthogonalEdgeComponent);

export const EDGE_TONES: EdgeTone[] = [
  "default",
  "critical",
  "blocked",
  "running",
  "traversed",
  "conditional",
];

/** Fixed-size (userSpaceOnUse) arrowheads, one per tone, so heads never scale with stroke width. */
export function ArrowMarkers({ prefix }: { prefix: string }) {
  return (
    <svg className="ag-defs" aria-hidden="true" focusable="false">
      <defs>
        {EDGE_TONES.map((tone) => (
          <marker
            key={tone}
            id={`${prefix}-${tone}`}
            className="ag-arrow"
            data-tone={tone}
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="9"
            markerHeight="9"
            markerUnits="userSpaceOnUse"
            orient="auto"
          >
            <path d="M0,1 L10,5 L0,9 z" />
          </marker>
        ))}
      </defs>
    </svg>
  );
}

export function edgeTone(edge: { critical?: boolean; blocked?: boolean; running?: boolean }): EdgeTone {
  if (edge.blocked) return "blocked";
  if (edge.running) return "running";
  if (edge.critical) return "critical";
  return "default";
}
