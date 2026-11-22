import { Fragment } from "react";

import type { Workbench, WorkbenchState } from "../state/workbench";
import { statusMeta, VIEWS, type View } from "../lib/labels";
import { StatCells } from "./ui";

const VIEW_LABELS: Record<View, string> = {
  topology: "工作流",
  design: "架构规格",
  matrix: "追踪矩阵",
  waterfall: "执行瀑布",
};

const STAGE_GLYPHS: Record<string, string> = { done: "✓", active: "▶", blocked: "✕", skipped: "—" };

export function StageHeader({ workbench, state }: { workbench: Workbench; state: WorkbenchState }) {
  const projection = state.projection;
  const runMeta = projection ? statusMeta(projection.status) : null;
  const connection = workbench.connectionText();
  const progress = projection?.progress;
  const paused = projection?.status === "paused";
  const hasTasks = Boolean(projection?.dag.nodes.length);
  const showPause = Boolean(projection && hasTasks && (projection.status === "running" || paused));
  const stages = projection?.stage_pipeline || [];
  return (
    <header className="stage-head">
      <div className="stage-title-row">
        <div className="stage-title-group">
          <h1 className="stage-title" id="run-title">
            {projection?.title || "Ariadne 研发编排控制台"}
          </h1>
          <span className="badge" data-variant={runMeta?.variant || "muted"} id="run-kicker">
            {runMeta ? `${runMeta.glyph} ${runMeta.label}` : "准备就绪"}
          </span>
          <span className="badge" data-variant="warn" id="run-connection" role="status" hidden={!connection}>
            {connection}
          </span>
        </div>
        <nav className="view-switcher" aria-label="视图切换">
          {VIEWS.map((view) => (
            <a
              key={view}
              className="rail-link"
              href={`#${view}`}
              data-view={view}
              aria-current={state.view === view ? "page" : undefined}
              onClick={(event) => {
                event.preventDefault();
                workbench.setView(view);
              }}
            >
              <span className="rail-label">{VIEW_LABELS[view]}</span>
            </a>
          ))}
        </nav>
        <div className="stage-actions">
          <button
            type="button"
            className="btn"
            id="btn-dock-open"
            hidden={state.dockOpen}
            onClick={() => workbench.setDockOpen(true, true)}
          >
            检查器
          </button>
          <button
            type="button"
            className={paused ? "btn btn-primary" : "btn btn-danger"}
            id="btn-toggle-pause"
            hidden={!showPause}
            onClick={() => void workbench.togglePause()}
          >
            {paused ? "恢复运行" : "暂停调度"}
          </button>
        </div>
      </div>
      <div className="stage-bar-row">
        <nav className="stage-pipeline" id="stage-pipeline" aria-label="阶段进度">
          {stages.map((stage, index) => (
            <Fragment key={stage.id || index}>
              <span className="stage-step" data-status={stage.status}>
                {`${STAGE_GLYPHS[stage.status] || "◷"} ${stage.label}`}
              </span>
              {index < stages.length - 1 ? <span className="stage-step-arrow">→</span> : null}
            </Fragment>
          ))}
        </nav>
        <div className="stat-band" id="stat-band" aria-label="运行指标">
          {progress && progress.total_tasks > 0 ? (
            <StatCells
              items={[
                {
                  label: "已验收任务",
                  value: `${progress.verified_tasks} / ${progress.total_tasks} · ${Math.floor(
                    (progress.verified_tasks * 100) / progress.total_tasks,
                  )}%`,
                },
              ]}
            />
          ) : null}
        </div>
      </div>
    </header>
  );
}
