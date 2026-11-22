/**
 * Public JSON contract between the vanilla Ariadne console (host) and the Graph Island.
 *
 * The host owns domain wording (status labels, localisation); the island owns geometry,
 * routing and visual channels. Nothing here references Olympus backend internals.
 */

export type DagTone = "ok" | "live" | "bad" | "warn" | "muted" | "planned";

export type EdgeTone =
  | "default"
  | "critical"
  | "blocked"
  | "running"
  | "traversed"
  | "conditional";

export interface DagPill {
  text: string;
  tone?: "warn" | "live" | "muted";
}

export interface DagNodeView {
  /** Stable task identifier; rendered as `data-task-id`. */
  id: string;
  /** Wave index from the projection; cards of one wave share a column. */
  layer: number;
  /** Raw display status; rendered as `data-status`. */
  status: string;
  /** Semantic surface/border tone of the card. */
  tone: DagTone;
  /** Row 2 (`role · kind`, or a retry note). */
  meta: string;
  /** Row 3 status capsule text, glyph included (`✓ 已验收`). */
  statusText: string;
  /** Optional row 1 badge (`×2`, `⚠ 风险`, `重试中`). */
  pill?: DagPill | null;
  /** Accessible name of the card. */
  ariaLabel: string;
  /** Native tooltip (task title). */
  title?: string;
  /** Extra stable hooks kept from the SVG renderer. */
  issue?: string;
  risk?: boolean;
}

export interface DagEdgeView {
  source: string;
  target: string;
  critical?: boolean;
  blocked?: boolean;
  running?: boolean;
}

export interface DagGraphProps {
  runId: string;
  nodes: DagNodeView[];
  edges: DagEdgeView[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
  emptyMessage?: string;
  ariaLabel?: string;
}

/** Island-side classification of a workflow node (badge + card accent). */
export type WorkflowNodeKind = "gate" | "validator" | "stage";

export interface WorkflowNodeView {
  /** Workflow node name; rendered as `data-workflow-node` and the monospace row. */
  id: string;
  /** Localised card title (`需求对齐`). */
  label: string;
  /** Raw `graph.nodes[*].kind` (`human_gate`, `validate_plan`, `align_goal`, …). */
  kind?: string;
  /** Raw projection status; rendered as `data-status` (`pending`, `skipped`, …). */
  status: string;
  /** Row 3 status capsule text, glyph included (`— 已跳过`). */
  statusText: string;
  /** Accessible name of the card (the island appends the kind badge word). */
  ariaLabel: string;
  /** Native tooltip. */
  title?: string;
}

export interface WorkflowEdgeView {
  source: string;
  target: string;
  /** Router fan-out: only taken at runtime. */
  conditional?: boolean;
  /** Present in `projection.workflow.traversed_edges`. */
  traversed?: boolean;
}

export interface WorkflowGraphProps {
  runId: string;
  nodes: WorkflowNodeView[];
  edges: WorkflowEdgeView[];
  selectedId?: string | null;
  /** Node kept in view on initial fit when the workflow is wider than its frame. */
  focusId?: string | null;
  onSelect?: (id: string) => void;
  emptyMessage?: string;
  ariaLabel?: string;
}

export interface GraphIslandHandle<P> {
  update(props: P): void;
  fitView(): void;
  destroy(): void;
}

export interface AriadneGraphAPI {
  readonly version: string;
  mountDag(container: HTMLElement, initialProps: DagGraphProps): GraphIslandHandle<DagGraphProps>;
  mountWorkflow(
    container: HTMLElement,
    initialProps: WorkflowGraphProps,
  ): GraphIslandHandle<WorkflowGraphProps>;
}
