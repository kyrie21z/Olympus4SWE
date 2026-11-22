import type { Workbench, WorkbenchState } from "../state/workbench";
import { statusMeta, WORKFLOW_LABELS } from "../lib/labels";
import { Badge } from "./ui";

export function ThreadRail({ workbench, state }: { workbench: Workbench; state: WorkbenchState }) {
  const collapsed = state.railCollapsed === true;
  const dark = state.theme === "dark";
  return (
    <nav id="rail" className={collapsed ? "scroller collapsed" : "scroller"} aria-label="工作台线程栏">
      <div className="rail-head-row">
        <div className="rail-brand">
          <span className="rail-mark" aria-hidden="true">
            ◈
          </span>
          <span className="rail-wordmark">Ariadne</span>
        </div>
        <div className="rail-head-actions">
          <button
            type="button"
            className="btn btn-quiet"
            id="theme-toggle"
            aria-label={dark ? "切换到浅色主题" : "切换到深色主题"}
            onClick={() => workbench.toggleTheme()}
          >
            {dark ? "☀" : "☾"}
          </button>
          <button
            type="button"
            className="btn btn-quiet"
            id="btn-rail-collapse"
            aria-expanded={collapsed ? "false" : "true"}
            aria-label="折叠侧栏"
            onClick={() => workbench.toggleRail()}
          >
            {collapsed ? "⇥" : "⇤"}
          </button>
        </div>
      </div>
      <button
        type="button"
        className="btn btn-primary rail-new-run"
        id="btn-new-run"
        onClick={() => void workbench.openNewRun()}
      >
        + 新建编排
      </button>
      <p className="rail-section-title">编排线程</p>
      <div className="thread-list scroller" id="thread-list" aria-label="编排线程列表">
        {state.runs.length ? (
          state.runs.map((run) => {
            const meta = statusMeta(run.status);
            return (
              <button
                type="button"
                className="thread-item"
                key={run.run_id}
                data-run-id={run.run_id}
                aria-pressed={run.run_id === state.runId ? "true" : "false"}
                onClick={() => workbench.selectRun(run.run_id)}
              >
                <span className="thread-item-title">{run.title}</span>
                <span className="thread-item-meta">
                  <Badge variant="muted">{WORKFLOW_LABELS[run.workflow_name] || run.workflow_name || "workflow"}</Badge>
                  <Badge variant={meta.variant}>{`${meta.glyph} ${meta.label}`}</Badge>
                </span>
              </button>
            );
          })
        ) : (
          <p className="thread-empty">尚无编排线程</p>
        )}
      </div>
      <div className="rail-foot" hidden>
        <label className="rail-foot-label" htmlFor="run-selector">
          运行实例
        </label>
        <select
          id="run-selector"
          className="run-select"
          aria-label="切换运行实例"
          value={state.runId || ""}
          onChange={(event) => workbench.selectRun(event.target.value)}
        >
          {state.runs.map((run) => (
            <option value={run.run_id} key={run.run_id}>
              {`${run.title} (${run.run_id})`}
            </option>
          ))}
        </select>
      </div>
    </nav>
  );
}
