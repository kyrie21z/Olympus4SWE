import { useLayoutEffect, useRef, type RefObject } from "react";

import { useRunStream } from "./api/useRunStream";
import { HitlDialog, NewRunDialog, Toast } from "./components/Dialogs";
import { HitlBanner } from "./components/HitlBanner";
import { InspectorDock } from "./components/InspectorDock";
import { ClarifyPanel, IntentCard } from "./components/IntentClarify";
import { StageHeader } from "./components/StageHeader";
import { ThreadRail } from "./components/ThreadRail";
import { TopologyView } from "./components/TopologyView";
import { DesignView, MatrixView, WaterfallView } from "./components/Views";
import { useWorkbench, type Workbench } from "./state/workbench";

const DOCK_MIN = 320;
const DOCK_MAX = 720;
const DOCK_STEP = 24;

function DockResizer({ cockpit }: { cockpit: RefObject<HTMLDivElement | null> }) {
  const drag = useRef<{ startX: number; startWidth: number } | null>(null);
  const dockWidth = () => document.getElementById("dock")?.getBoundingClientRect().width || 0;
  const setWidth = (width: number) =>
    cockpit.current?.style.setProperty(
      "--dock-width",
      `${Math.round(Math.min(Math.max(width, DOCK_MIN), DOCK_MAX))}px`,
    );
  return (
    <div
      id="dock-resizer"
      role="separator"
      aria-orientation="vertical"
      aria-label="调整检查器宽度"
      tabIndex={0}
      onPointerDown={(event) => {
        drag.current = { startX: event.clientX, startWidth: dockWidth() };
        event.currentTarget.setPointerCapture(event.pointerId);
        event.preventDefault();
      }}
      onPointerMove={(event) => {
        if (!drag.current) return;
        setWidth(drag.current.startWidth - (event.clientX - drag.current.startX));
      }}
      onPointerUp={(event) => {
        drag.current = null;
        event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onKeyDown={(event) => {
        const step = event.key === "ArrowLeft" ? DOCK_STEP : event.key === "ArrowRight" ? -DOCK_STEP : 0;
        if (!step) return;
        event.preventDefault();
        setWidth(dockWidth() + step);
      }}
    />
  );
}

export function App({ workbench }: { workbench: Workbench }) {
  const state = useWorkbench(workbench);
  const cockpit = useRef<HTMLDivElement>(null);

  useRunStream(
    state.runId,
    state.streamGeneration,
    {
      current: () => workbench.state.projection,
      apply: workbench.applyProjection,
      phase: (phase) => workbench.setPhase(phase),
      activity: (previews) => workbench.setActivity(previews),
      missing: (message) => workbench.toast(message),
    },
    { sse: workbench.sse },
  );

  useLayoutEffect(() => {
    document.body.dataset.dock = state.dockOpen ? "open" : "closed";
    if (state.railCollapsed !== null) {
      document.body.dataset.rail = state.railCollapsed ? "collapsed" : "expanded";
    }
  });

  return (
    <>
      <div className="cockpit" ref={cockpit}>
        <ThreadRail workbench={workbench} state={state} />
        <main id="stage" className="scroller">
          <StageHeader workbench={workbench} state={state} />
          <div className="stage-inline">
            <HitlBanner workbench={workbench} state={state} />
            <IntentCard state={state} />
            <ClarifyPanel workbench={workbench} state={state} />
          </div>
          <TopologyView workbench={workbench} state={state} />
          <DesignView state={state} />
          <MatrixView workbench={workbench} state={state} />
          <WaterfallView state={state} />
          <footer className="stage-foot">
            <span id="foot-note">
              {state.projection
                ? `${state.projection.run_id} · 最近事件 #${state.projection.latest_event_seq}`
                : "Ariadne M4"}
            </span>
          </footer>
        </main>
        <DockResizer cockpit={cockpit} />
        <InspectorDock workbench={workbench} state={state} />
      </div>
      <NewRunDialog workbench={workbench} state={state} />
      <HitlDialog workbench={workbench} state={state} />
      <Toast state={state} />
    </>
  );
}
