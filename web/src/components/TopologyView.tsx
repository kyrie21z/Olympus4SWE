import type { RunProjection } from "../api/types";
import { dagGraphProps, workflowGraphProps } from "../lib/graphProps";
import {
  currentStage,
  DELIVERY_STATUS,
  formatTokens,
  NODE_LABELS,
  OUTPUT_ACTIONS,
  progressText,
  statusMeta,
  taskDisplayStatus,
  workflowCurrentNode,
  WORKFLOW_LABELS,
} from "../lib/labels";
import type { Workbench, WorkbenchState } from "../state/workbench";
import { RecordedArtifactCards } from "./Artifacts";
import { DagGraphHost, WorkflowGraphHost } from "./GraphHost";
import { Badge, StatCells } from "./ui";

function StageLivePanel({ workbench, state, projection }: { workbench: Workbench; state: WorkbenchState; projection: RunProjection | null }) {
  if (!projection) {
    return <section className="stage-live-panel card" id="stage-live-panel" hidden aria-labelledby="run-progress-title" />;
  }
  const stage = currentStage(projection);
  const recent = projection.recent_progress;
  const stamp = recent?.created_at || projection.created_at;
  const outputs = projection.key_outputs || [];
  const latest = outputs[0];
  const budget = projection.budget;
  return (
    <section className="stage-live-panel card" id="stage-live-panel" aria-labelledby="run-progress-title">
      <div className="card-head">
        <h2 className="card-title" id="run-progress-title">
          运行进展
        </h2>
        <details className="run-detail">
          <summary>运行详情</summary>
          <div id="run-detail-stats" className="stat-band">
            <StatCells
              items={[
                {
                  label: "LLM",
                  value: budget.max_calls == null ? `${budget.calls_used} 次` : `${budget.calls_used} / ${budget.max_calls}`,
                },
                { label: "Token", value: formatTokens(budget.total_tokens) },
                { label: "修订", value: `r${projection.state_revision}` },
                { label: "最近事件", value: `#${projection.latest_event_seq}` },
              ]}
            />
          </div>
        </details>
      </div>
      <p id="run-current-stage" className="stage-activity-item" role="status">
        {`当前阶段 · ${stage.label}`}
      </p>
      <p className="run-progress-note">
        <span id="run-recent-progress">{`最近确认 · ${progressText(recent)}`}</span>{" "}
        <time id="run-progress-time" dateTime={stamp}>
          {stamp ? new Date(stamp).toLocaleString("zh-CN") : ""}
        </time>
      </p>
      <div id="run-key-output" className="stage-activity-item" hidden={!latest}>
        <p id="run-output-summary" role="status">
          {latest?.summary || ""}
        </p>
        <button
          id="run-output-detail"
          type="button"
          className="btn"
          hidden={!latest}
          onClick={() => void workbench.openOutput(workbench.state.projection?.key_outputs?.[0])}
        >
          {latest ? (latest.current || latest.detail_available ? OUTPUT_ACTIONS[latest.kind] : "查看版本记录") : ""}
        </button>
      </div>
      <details id="run-output-history" className="run-detail" hidden={!outputs.length}>
        <summary>产物历史</summary>
        <ol id="run-output-versions">
          {outputs.map((output) => {
            const detail = state.attemptDetails[output.key];
            return (
              <li key={output.key} data-output-key={output.key} tabIndex={-1}>
                <span>{output.summary}</span>
                <code>
                  {output.content_hash
                    ? `${output.artifact_id} · ${output.content_hash}`
                    : new Date(output.created_at).toLocaleString()}
                </code>
                {output.current || output.detail_available ? (
                  <button
                    type="button"
                    className="btn"
                    data-history-output={output.key}
                    onClick={() => void workbench.openOutput(output)}
                  >
                    {OUTPUT_ACTIONS[output.kind]}
                  </button>
                ) : null}
                {detail ? (
                  <section className="run-attempt-detail" tabIndex={-1} aria-label={output.summary}>
                    {detail.status === "loading" ? (
                      "正在读取执行产物…"
                    ) : (
                      <RecordedArtifactCards
                        artifacts={detail.artifacts}
                        manifest={detail.artifacts.manifest_md}
                        showEmpty={false}
                      />
                    )}
                  </section>
                ) : null}
              </li>
            );
          })}
        </ol>
      </details>
    </section>
  );
}

function workflowBadge(projection: RunProjection | null): string {
  if (!projection) return "拓扑加载中…";
  const workflow = projection.workflow;
  const name = workflow.name || projection.workflow_name || "workflow";
  const label = WORKFLOW_LABELS[name] || name;
  if (!workflow.graph?.nodes?.length) return `${label} · 拓扑不可用`;
  const current = workflowCurrentNode(workflow);
  const currentText = current ? `${NODE_LABELS[current] || current} (${current})` : "";
  return `${label} · ${workflow.topology || "topology"} · stop=${workflow.stop_after || "-"}${
    currentText ? ` · 当前 ${currentText}` : ""
  }`;
}

function DeliveryAcceptanceSection({ workbench, projection }: { workbench: Workbench; projection: RunProjection | null }) {
  const review = projection?.delivery_acceptance;
  const hidden = !review || !review.criteria.length;
  return (
    <section className="delivery-acceptance" id="delivery-acceptance" aria-labelledby="delivery-title" hidden={hidden}>
      <header className="view-head">
        <div className="view-head-meta">
          <h2 className="view-title" id="delivery-title">
            交付验收
          </h2>
          <span className="card-note" id="delivery-spec">
            {review && !hidden ? `GoalSpec v${review.spec_version}` : ""}
          </span>
        </div>
        <div className="stage-pills" id="delivery-counts" aria-label="验收状态汇总">
          {review && !hidden
            ? Object.entries(DELIVERY_STATUS).map(([status, [label, variant]]) => (
                <Badge variant={variant} key={status}>
                  {`${label} ${review.counts[status as keyof typeof review.counts]}`}
                </Badge>
              ))
            : null}
        </div>
      </header>
      <ul className="delivery-criteria card" id="delivery-criteria" aria-label="逐条验收条件">
        {review && !hidden
          ? review.criteria.map((row, index) => {
              const [label, variant] = DELIVERY_STATUS[row.status] || DELIVERY_STATUS.unverified;
              const ref = `${row.requirement_id} · 条件 ${row.criterion_index}`;
              return (
                <li className="delivery-criterion" key={`${ref}-${index}`}>
                  <div>
                    <span className="card-note">{ref}</span>
                    <p>{row.text}</p>
                    {row.reason ? <p className="card-note">{row.reason}</p> : null}
                  </div>
                  <div className="delivery-result">
                    <Badge variant={row.risk_accepted ? "warn" : variant}>{label}</Badge>
                    {row.evidence ? (
                      <button
                        type="button"
                        className="btn btn-quiet"
                        data-delivery-evidence={index}
                        aria-controls="dock"
                        aria-label={`查看证据 · ${ref}`}
                        onClick={() => workbench.showDeliveryEvidence(index)}
                      >
                        查看证据
                      </button>
                    ) : null}
                  </div>
                </li>
              );
            })
          : null}
      </ul>
    </section>
  );
}

function Swimlanes({ workbench, state, projection }: { workbench: Workbench; state: WorkbenchState; projection: RunProjection | null }) {
  const lanes = projection?.swimlanes || [];
  return (
    <div className="lanes" id="swimlanes">
      {lanes.map((lane) => (
        <div className="lane" key={lane.actor}>
          <span className="lane-actor">{lane.actor}</span>
          <div className="lane-items">
            {(lane.items || []).map((item) => {
              const displayStatus = taskDisplayStatus(item, projection);
              const meta = statusMeta(displayStatus);
              return (
                <button
                  type="button"
                  className="lane-chip"
                  key={item.id}
                  data-task-id={item.task_id || ""}
                  data-status={displayStatus}
                  aria-pressed={item.task_id && item.task_id === state.selectedTaskId ? "true" : "false"}
                  onClick={() => item.task_id && workbench.selectTask(item.task_id)}
                >
                  {`${meta.glyph} ${item.label}`}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

export function TopologyView({ workbench, state }: { workbench: Workbench; state: WorkbenchState }) {
  const projection = state.projection;
  const hasDag = Boolean(projection?.dag.nodes.length);
  const awaitingPlanApproval = projection?.hitl_banner?.kind === "plan_gate";
  const dagHidden =
    !hasDag || (awaitingPlanApproval && state.openedPlanGraphHash !== projection?.plan?.content_hash);
  const approvedPlan = projection?.plan_approved ? projection.plan : null;
  const lanes = projection?.swimlanes || [];
  return (
    <section
      className="view"
      data-pane="topology"
      id="view-topology"
      aria-labelledby="workflow-title"
      hidden={state.view !== "topology"}
    >
      <StageLivePanel workbench={workbench} state={state} projection={projection} />

      <section className="workflow-section" aria-labelledby="workflow-title">
        <header className="view-head">
          <div className="view-head-meta">
            <h2 className="view-title" id="workflow-title">
              编排工作流
            </h2>
            <span className="card-note" id="workflow-badge">
              {workflowBadge(projection)}
            </span>
          </div>
          <div className="dag-legend" aria-label="状态图例">
            <span className="legend-item">✓ 完成</span>
            <span className="legend-item">▶ 运行中</span>
            <span className="legend-item">✕ 中断/阻塞</span>
            <span className="legend-item">◷ 待确认</span>
            <span className="legend-item">— 已跳过</span>
            <span className="legend-item">
              <span className="legend-swatch" data-tone="traversed" aria-hidden="true" /> 已执行路径
            </span>
            <span className="legend-item">
              <span className="legend-dash" aria-hidden="true" /> 未走条件路由 / 修复回边
            </span>
          </div>
        </header>
        {projection ? (
          <WorkflowGraphHost {...workflowGraphProps(projection, state.workflowNode, workbench.selectWorkflowNode)} />
        ) : (
          <div className="graph-host" id="workflow-graph" data-graph-host="workflow" />
        )}
      </section>

      <section className="task-dag-section" id="task-dag-section" aria-labelledby="topology-title" hidden={dagHidden}>
        <header className="view-head">
          <h2 className="view-title" id="topology-title" tabIndex={-1}>
            任务依赖图
          </h2>
          <div className="dag-legend" aria-label="任务图例">
            <span className="legend-item">✓ 已验收</span>
            <span className="legend-item">▶ 运行中</span>
            <span className="legend-item">✕ 阻塞</span>
            <span className="legend-item">⚠ 人工接受</span>
            <span className="legend-item">
              <span className="legend-swatch" data-tone="critical" aria-hidden="true" /> 关键路径
            </span>
            <span className="legend-item">
              <span className="legend-swatch" data-tone="blocked" aria-hidden="true" /> 阻塞传递
            </span>
            <span className="legend-item">
              <span className="legend-swatch" data-tone="running" aria-hidden="true" /> 执行中
            </span>
            <span className="legend-item">
              <span className="legend-ring" aria-hidden="true" /> 当前选中
            </span>
          </div>
        </header>
        {projection ? (
          <DagGraphHost {...dagGraphProps(projection, state.selectedTaskId, workbench.selectTask)} />
        ) : (
          <div className="graph-host" id="dag-graph" data-graph-host="dag" />
        )}
        <details id="run-plan-detail" className="run-detail" hidden={!approvedPlan}>
          <summary>{approvedPlan ? `查看完整任务计划 · v${approvedPlan.version}` : "查看完整任务计划"}</summary>
          <pre className="code-block scroller" tabIndex={0}>
            {approvedPlan ? JSON.stringify(approvedPlan, null, 2) : ""}
          </pre>
        </details>
      </section>

      <DeliveryAcceptanceSection workbench={workbench} projection={projection} />

      <section className="lane-section" id="lane-section" aria-labelledby="lane-title" hidden={!hasDag || awaitingPlanApproval}>
        <div className="card-head">
          <h3 className="card-title" id="lane-title">
            Agent 泳道（时间 →）
          </h3>
          <span className="card-note" id="lane-badge">
            {projection ? `${lanes.length} 隔离通道` : "隔离通道"}
          </span>
        </div>
        <Swimlanes workbench={workbench} state={state} projection={projection} />
      </section>
    </section>
  );
}
