import { Fragment, type CSSProperties } from "react";

import { statusMeta } from "../lib/labels";
import type { Workbench, WorkbenchState } from "../state/workbench";
import { DesignSpecCard, GoalSpecCard } from "./spec";
import { Badge, Notice } from "./ui";

export function DesignView({ state }: { state: WorkbenchState }) {
  const projection = state.projection;
  const spec = projection?.spec;
  const design = projection?.design;
  return (
    <section className="view" data-pane="design" id="view-design" aria-labelledby="design-title" hidden={state.view !== "design"}>
      <header className="view-head">
        <h2 className="view-title" id="design-title">
          架构规格
        </h2>
        <Badge
          id="design-badge"
          variant={design && projection?.design_approved ? "ok" : spec ? "warn" : "muted"}
        >
          {design ? `DesignSpec v${design.version || 1}` : spec ? `GoalSpec v${spec.version || 1}` : "等待规格"}
        </Badge>
      </header>
      <div className="design-grid" id="design-body">
        {!projection ? null : !spec && !design ? (
          <Notice level="info">
            当前运行尚未生成 GoalSpec / DesignSpec。先新建编排提交需求，或批准当前关口后继续推进。
          </Notice>
        ) : (
          <>
            {spec ? <GoalSpecCard spec={spec} projection={projection} /> : null}
            {design ? <DesignSpecCard design={design} projection={projection} /> : null}
          </>
        )}
      </div>
    </section>
  );
}

export function MatrixView({ workbench, state }: { workbench: Workbench; state: WorkbenchState }) {
  const rows = state.projection?.traceability_matrix || [];
  const counts = { verified: 0, running: 0, failed: 0, risk: 0, pending: 0 };
  rows.forEach((row) => {
    if (row.verification_status === "verified") counts.verified += 1;
    else if (row.verification_status === "running") counts.running += 1;
    else if (row.verification_status === "failed") counts.failed += 1;
    else if (row.verification_status === "unverified_risk_accepted") counts.risk += 1;
    else counts.pending += 1;
  });
  return (
    <section className="view" data-pane="matrix" id="view-matrix" aria-labelledby="matrix-title" hidden={state.view !== "matrix"}>
      <header className="view-head">
        <h2 className="view-title" id="matrix-title">
          需求追踪矩阵
        </h2>
        <div className="stage-pills" id="fr-strip" aria-label="需求验证状态汇总">
          {counts.verified ? <Badge variant="ok">{`✓ 已验证 ${counts.verified}`}</Badge> : null}
          {counts.running ? <Badge variant="live">{`▶ 验证中 ${counts.running}`}</Badge> : null}
          {counts.failed ? <Badge variant="bad">{`✕ 失败 ${counts.failed}`}</Badge> : null}
          {counts.risk ? <Badge variant="warn">{`⚠ 人工接受 ${counts.risk}`}</Badge> : null}
          {counts.pending ? <Badge variant="muted">{`◷ 待验证 ${counts.pending}`}</Badge> : null}
        </div>
      </header>
      <div className="table-wrapper scroller" tabIndex={0} role="region" aria-labelledby="matrix-caption">
        <table className="matrix-table">
          <caption id="matrix-caption">需求、设计模块、任务与验收结果四级追踪表</caption>
          <thead>
            <tr>
              <th scope="col">需求编号</th>
              <th scope="col">来源与需求描述</th>
              <th scope="col">设计模块</th>
              <th scope="col">承接任务</th>
              <th scope="col">验收状态与证据</th>
            </tr>
          </thead>
          <tbody id="matrix-tbody">
            {rows.map((row) => {
              const meta = statusMeta(row.verification_status);
              const modules = row.design_modules || [];
              const tasks = row.task_ids || [];
              return (
                <tr key={row.requirement_id}>
                  <th scope="row">{row.requirement_id}</th>
                  <td>
                    <strong>{row.source_ref}</strong>
                    {` · ${row.description}`}
                  </td>
                  <td>
                    {modules.length
                      ? modules.map((module, index) => (
                          <Fragment key={module.id}>
                            {index ? " " : null}
                            <button
                              type="button"
                              className="lane-chip"
                              data-jump-design={module.id}
                              title={module.name || ""}
                              onClick={() => workbench.setView("design")}
                            >
                              {module.id}
                            </button>
                          </Fragment>
                        ))
                      : "—"}
                  </td>
                  <td>
                    {tasks.length
                      ? tasks.map((taskId, index) => (
                          <Fragment key={taskId}>
                            {index ? " " : null}
                            <button
                              type="button"
                              className="lane-chip"
                              data-jump-task={taskId}
                              onClick={() => workbench.selectTask(taskId)}
                            >
                              {taskId}
                            </button>
                          </Fragment>
                        ))
                      : "—"}
                  </td>
                  <td>
                    <Badge variant={meta.variant}>{`${meta.glyph} ${row.verification_label}`}</Badge>{" "}
                    <span className="card-note">{row.evidence_note}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function WaterfallView({ state }: { state: WorkbenchState }) {
  const turns = state.projection?.waterfall_turns || [];
  return (
    <section
      className="view"
      data-pane="waterfall"
      id="view-waterfall"
      aria-labelledby="waterfall-title"
      hidden={state.view !== "waterfall"}
    >
      <header className="view-head">
        <h2 className="view-title" id="waterfall-title">
          Turn 瀑布
        </h2>
        <Badge id="waterfall-count">{`${turns.length} turns`}</Badge>
      </header>
      <div className="waterfall" id="waterfall-list">
        {turns.map((turn) => {
          const meta = statusMeta(turn.status);
          const offset = Math.min(90, Math.max(0, Number(turn.offset_pct) || 0));
          const width = Math.min(100 - offset, Math.max(8, Number(turn.width_pct) || 8));
          const barStyle = { "--wf-offset": `${offset}%`, "--wf-fill": `${width}%` } as CSSProperties;
          return (
            <article className="wf-row" data-status={turn.status} key={turn.turn_id}>
              <div className="wf-actor">
                <span>{turn.actor}</span>
                <Badge variant={meta.variant}>{meta.label}</Badge>
              </div>
              <div className="wf-main">
                <span className="wf-title">{turn.title}</span>
                <span className="wf-summary">{turn.summary}</span>
                <span className="wf-track">
                  <span className="wf-bar" style={barStyle} />
                </span>
                <span className="wf-summary">{turn.receipt ? turn.receipt.one_line : ""}</span>
                {turn.manifest_md ? (
                  <div className="tool-list">
                    <details className="tool-row">
                      <summary>
                        <span className="tool-mark">▤</span>
                        <span className="tool-name">MANIFEST.md</span>
                        <span className="tool-target">{turn.title}</span>
                        <span className="tool-duration" />
                      </summary>
                      <pre className="tool-body">{turn.manifest_md}</pre>
                    </details>
                  </div>
                ) : null}
              </div>
              <div className="wf-meta">
                <span>{`${turn.duration_ms} ms`}</span>
                <span>{`${turn.steps} steps`}</span>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
