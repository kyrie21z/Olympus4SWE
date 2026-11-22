import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  type ReactNode,
  type RefObject,
} from "react";
import {
  ReactFlow,
  useReactFlow,
  type Edge,
  type EdgeTypes,
  type Node,
  type NodeTypes,
} from "@xyflow/react";

import { ArrowMarkers } from "../edges/OrthogonalEdge";
import type { Point } from "../geometry";
import type { IslandController } from "../island";

/** Viewport padding (px) shared by every graph frame. */
export const PAD = 24;
/**
 * Initial auto-fit never magnifies and, by default, never shrinks either: both Topology graphs
 * open at the same crisp 1:1 card scale, and a graph larger than its frame pans instead.
 */
export const DEFAULT_MIN_FIT_ZOOM = 1;
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 2;
const ZOOM_STEP = 1.2;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

/** What the canvas needs to know about a finished layout to frame it. */
export interface CanvasLayout {
  signature: string;
  mode: string;
  width: number;
  height: number;
}

/** Stable, CSS-safe id prefix for one canvas's arrowhead markers. */
export function useMarkerPrefix(): string {
  return `ag-arrow-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
}

/** Frame height that follows the laid-out graph within [min, max]. */
export function frameHeight(graphHeight: number, min: number, max: number): number {
  return Math.round(Math.min(max, Math.max(min, graphHeight + PAD * 2)));
}

/** Center when the graph fits; otherwise keep `focus` in view, clamped to the graph's edges. */
function axisOffset(size: number, extent: number, focus: number | null, zoom: number): number {
  if (extent + PAD * 2 <= size) return (size - extent) / 2;
  if (focus === null) return PAD;
  return Math.min(PAD, Math.max(size - PAD - extent, size / 2 - focus * zoom));
}

/**
 * Centers the laid-out graph at 1:1 when it fits; otherwise scales down no further than minZoom
 * and keeps the focus point in view. Re-fits only on topology change, or on container resize
 * before the user has moved the viewport.
 */
function AutoFit({
  layout,
  rootRef,
  controller,
  userMoved,
  minZoom,
  focusRef,
}: {
  layout: CanvasLayout | null;
  rootRef: RefObject<HTMLDivElement | null>;
  controller: IslandController;
  userMoved: RefObject<boolean>;
  minZoom: number;
  focusRef: RefObject<Point | null>;
}) {
  const flow = useReactFlow();
  const fitted = useRef<string | null>(null);

  const fit = useCallback((): boolean => {
    const root = rootRef.current;
    if (!layout || !root || !flow.viewportInitialized) return false;
    const width = root.clientWidth;
    const height = root.clientHeight;
    if (!width || !height) return false;
    const zoom = Math.max(
      minZoom,
      Math.min(1, (width - PAD * 2) / layout.width, (height - PAD * 2) / layout.height),
    );
    const focus = focusRef.current;
    const x = axisOffset(width, layout.width * zoom, focus ? focus.x : null, zoom);
    const y = axisOffset(height, layout.height * zoom, focus ? focus.y : null, zoom);
    void flow.setViewport({ x: Math.round(x), y: Math.round(y), zoom });
    fitted.current = layout.signature;
    userMoved.current = false;
    return true;
  }, [flow, focusRef, layout, minZoom, rootRef, userMoved]);

  controller.fit = () => {
    fit();
  };

  useLayoutEffect(() => {
    if (layout && fitted.current !== layout.signature) fit();
  });

  useEffect(() => {
    const root = rootRef.current;
    if (!root || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => {
      if (!layout) return;
      if (fitted.current !== layout.signature || !userMoved.current) fit();
    });
    observer.observe(root);
    return () => observer.disconnect();
  }, [fit, layout, rootRef, userMoved]);

  return null;
}

/**
 * Scroll-safe viewport navigation: unmodified wheel scrolls the host page (#stage),
 * Ctrl/Cmd + wheel zooms around the pointer, and a compact corner toolbar exposes +, -,
 * fit-all ("适应画布"), and 100% reset controls.
 */
function ViewportControls({
  layout,
  rootRef,
  userMoved,
  focusRef,
}: {
  layout: CanvasLayout | null;
  rootRef: RefObject<HTMLDivElement | null>;
  userMoved: RefObject<boolean>;
  focusRef: RefObject<Point | null>;
}) {
  const flow = useReactFlow();

  const zoomAt = useCallback(
    (nextZoomRaw: number, anchorX: number, anchorY: number) => {
      if (!flow.viewportInitialized) return;
      const current = flow.getViewport();
      const nextZoom = clamp(nextZoomRaw, MIN_ZOOM, MAX_ZOOM);
      if (Math.abs(nextZoom - current.zoom) < 0.0005) return;
      const ratio = nextZoom / current.zoom;
      const x = anchorX - (anchorX - current.x) * ratio;
      const y = anchorY - (anchorY - current.y) * ratio;
      userMoved.current = true;
      void flow.setViewport({ x: Math.round(x), y: Math.round(y), zoom: nextZoom });
    },
    [flow, userMoved],
  );

  const stepZoom = useCallback(
    (factor: number) => {
      const root = rootRef.current;
      if (!root || !flow.viewportInitialized) return;
      const current = flow.getViewport();
      zoomAt(current.zoom * factor, root.clientWidth / 2, root.clientHeight / 2);
    },
    [flow, rootRef, zoomAt],
  );

  const fitAll = useCallback(() => {
    const root = rootRef.current;
    if (!layout || !root || !flow.viewportInitialized) return;
    const width = root.clientWidth;
    const height = root.clientHeight;
    if (!width || !height || !layout.width || !layout.height) return;
    const zoom = clamp(
      Math.min(1, (width - PAD * 2) / layout.width, (height - PAD * 2) / layout.height),
      MIN_ZOOM,
      1,
    );
    const x = (width - layout.width * zoom) / 2;
    const y = (height - layout.height * zoom) / 2;
    userMoved.current = true;
    void flow.setViewport({ x: Math.round(x), y: Math.round(y), zoom });
  }, [flow, layout, rootRef, userMoved]);

  const resetOneToOne = useCallback(() => {
    const root = rootRef.current;
    if (!layout || !root || !flow.viewportInitialized) return;
    const width = root.clientWidth;
    const height = root.clientHeight;
    if (!width || !height) return;
    const focus = focusRef.current;
    const x = axisOffset(width, layout.width, focus ? focus.x : null, 1);
    const y = axisOffset(height, layout.height, focus ? focus.y : null, 1);
    userMoved.current = true;
    void flow.setViewport({ x: Math.round(x), y: Math.round(y), zoom: 1 });
  }, [flow, focusRef, layout, rootRef, userMoved]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return undefined;
    const onWheel = (event: WheelEvent) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      if (!flow.viewportInitialized) return;
      const rect = root.getBoundingClientRect();
      const delta = clamp(-event.deltaY, -180, 180);
      const factor = Math.exp(delta * 0.0025);
      const current = flow.getViewport();
      zoomAt(current.zoom * factor, event.clientX - rect.left, event.clientY - rect.top);
    };
    root.addEventListener("wheel", onWheel, { passive: false });
    return () => root.removeEventListener("wheel", onWheel);
  }, [flow, rootRef, zoomAt]);

  return (
    <div className="ag-toolbar nopan nodrag" role="toolbar" aria-label="画布视图控制">
      <button
        type="button"
        className="ag-tool-btn"
        data-graph-control="zoom-out"
        aria-label="缩小"
        title="缩小"
        onClick={() => stepZoom(1 / ZOOM_STEP)}
      >
        −
      </button>
      <button
        type="button"
        className="ag-tool-btn"
        data-graph-control="zoom-in"
        aria-label="放大"
        title="放大"
        onClick={() => stepZoom(ZOOM_STEP)}
      >
        +
      </button>
      <button
        type="button"
        className="ag-tool-btn ag-tool-text"
        data-graph-control="fit"
        aria-label="适应画布"
        title="适应画布"
        onClick={fitAll}
      >
        适应画布
      </button>
      <button
        type="button"
        className="ag-tool-btn ag-tool-text"
        data-graph-control="reset"
        aria-label="重置 100%"
        title="重置 100%"
        onClick={resetOneToOne}
      >
        100%
      </button>
    </div>
  );
}

export interface GraphCanvasProps<N extends Node, E extends Edge> {
  /** `data-graph` hook (`dag` | `workflow`). */
  graph: string;
  layout: CanvasLayout | null;
  height: number;
  ariaLabel: string;
  nodes: N[];
  edges: E[];
  nodeTypes: NodeTypes;
  edgeTypes: EdgeTypes;
  markerPrefix: string;
  controller: IslandController;
  /** Lower bound of the initial auto-fit zoom (default 1); the upper bound is always 1. */
  minFitZoom?: number;
  /** Graph-space point kept in view when the graph overflows the frame on initial fit. */
  focus?: Point | null;
  /** Extra viewport content (e.g. Wave headers in a ViewportPortal). */
  children?: ReactNode;
}

/**
 * The one read-only, scroll-safe React Flow canvas both Topology graphs share: fixed-size
 * tone-matched markers, non-draggable cards, page-scrolling wheel, drag-to-pan and auto-fit.
 */
export function GraphCanvas<N extends Node, E extends Edge>(props: GraphCanvasProps<N, E>) {
  const { layout, controller } = props;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const userMoved = useRef(false);
  const focusRef = useRef<Point | null>(props.focus ?? null);
  focusRef.current = props.focus ?? null;

  return (
    <div
      ref={rootRef}
      className="ag-island"
      data-graph={props.graph}
      data-layout={layout ? layout.mode : "pending"}
      role="group"
      aria-label={props.ariaLabel}
      style={{ height: props.height }}
    >
      <ArrowMarkers prefix={props.markerPrefix} />
      <ReactFlow<N, E>
        nodes={props.nodes}
        edges={props.edges}
        nodeTypes={props.nodeTypes}
        edgeTypes={props.edgeTypes}
        nodesDraggable={false}
        nodesConnectable={false}
        nodesFocusable={false}
        edgesFocusable={false}
        elementsSelectable={false}
        disableKeyboardA11y
        deleteKeyCode={null}
        selectionKeyCode={null}
        multiSelectionKeyCode={null}
        zoomOnScroll={false}
        panOnScroll={false}
        zoomOnPinch
        zoomOnDoubleClick={false}
        panOnDrag
        preventScrolling={false}
        minZoom={MIN_ZOOM}
        maxZoom={MAX_ZOOM}
        proOptions={{ hideAttribution: true }}
        onMoveStart={(event) => {
          if (event) userMoved.current = true;
        }}
      >
        <AutoFit
          layout={layout}
          rootRef={rootRef}
          controller={controller}
          userMoved={userMoved}
          minZoom={props.minFitZoom ?? DEFAULT_MIN_FIT_ZOOM}
          focusRef={focusRef}
        />
        <ViewportControls
          layout={layout}
          rootRef={rootRef}
          userMoved={userMoved}
          focusRef={focusRef}
        />
        {props.children}
      </ReactFlow>
    </div>
  );
}
