import type { ReactNode } from "react";

import type { DagNode, DeliveryAcceptance, DeliveryCriterion, RunProjection } from "../api/types";
import { deliveryLogUrl } from "../api/client";
import {
  DELIVERY_STATUS,
  localizeError,
  NODE_LABELS,
  RETRY_MESSAGE,
  RETRY_NEXT_STEP,
  statusMeta,
  taskDisplayStatus,
  type Tab,
} from "../lib/labels";
import type { Workbench, WorkbenchState } from "../state/workbench";
import { RecordedArtifactCards } from "./Artifacts";
import { Badge, Card, KeyValues, Notice, ReceiptBlock, Row, SubPane, ToolRow } from "./ui";

interface DockContent {
  title: string;
  stages: ReactNode;
  subpane?: ReactNode;
  tools?: ReactNode;
  tabs?: ReactNode;
  body: ReactNode;
  receipt?: ReactNode;
}

const TAB_LABELS: Record<Tab, string> = { overview: "概览", artifacts: "产物与审查", events: "事件日志" };

function Tabs({ workbench, active }: { workbench: Workbench; active: Tab }) {
  return (
    <div className="tabs" role="tablist" aria-label="任务详情视图切换">
      {(Object.keys(TAB_LABELS) as Tab[]).map((tab) => (
        <button
          type="button"
          role="tab"
          className="tab"
          key={tab}
          data-tab={tab}
          aria-selected={active === tab ? "true" : "false"}
          onClick={() => workbench.setTab(tab)}
        >
          {TAB_LABELS[tab]}
        </button>
      ))}
    </div>
  );
}

function EventRows({ events }: { events: RunProjection["events"] }) {
  return (
    <>
      {events.map((event) => (
        <div className="event-row" key={event.event_seq}>
          <span>{`#${event.event_seq}`}</span>
          <span className="event-type">{event.event_type}</span>
          <span className="event-payload">{`${event.actor} · r${event.state_revision}`}</span>
        </div>
      ))}
    </>
  );
}

const WATCHDOG_LABELS: Record<string, [string, "ok" | "warn" | "bad"]> = {
  HEALTHY: ["进展正常", "ok"],
  STUCK_LOOP: ["疑似循环", "warn"],
  SCOPE_DRIFT: ["偏离范围", "warn"],
  BLOCKED_EXTERNAL: ["外部阻塞", "bad"],
};

const TOOL_STATUS_LABELS: Record<string, string> = { ok: "成功", error: "出错", denied: "已拦截" };

function activityPreview(state: WorkbenchState, task: DagNode): string {
  if (state.projection?.status !== "running" || task.status !== "running") return "";
  const previews = state.activity;
  if (!previews) return task.live_progress?.preview || "";
  const entry = previews[task.task_id];
  return entry?.attempt === task.attempt ? entry.preview : "";
}

function LiveProgressCard({ state, task }: { state: WorkbenchState; task: DagNode }) {
  const progress = task.live_progress;
  if (!progress?.driver_name && !progress?.last_event_at) return null;
  const current = progress.current_tool;
  const verdict = progress.watchdog_verdict ? WATCHDOG_LABELS[progress.watchdog_verdict] : undefined;
  const hasDiagnosis = Boolean(verdict || progress.watchdog_summary);
  const preview = activityPreview(state, task);
  return (
    <Card title={task.status === "running" ? "实时执行" : "最近执行"}>
      <KeyValues>
        {progress.driver_name ? (
          <Row label="驱动" mono>
            {progress.driver_name}
          </Row>
        ) : null}
        {current ? (
          <Row label="当前工具" mono>
            {`${current.tool} · ${current.target_hint}`}
          </Row>
        ) : null}
        {progress.recent_tools
          .slice()
          .reverse()
          .map((entry, index) => (
            <Row label={entry.tool} mono key={`${entry.step}-${index}`}>
              {`${TOOL_STATUS_LABELS[entry.status]} · ${entry.duration_ms} ms`}
            </Row>
          ))}
        {hasDiagnosis ? (
          <Row label="Watchdog">
            {verdict ? <Badge variant={verdict[1]}>{verdict[0]}</Badge> : null}
            {verdict && progress.watchdog_summary ? " " : null}
            {progress.watchdog_summary ? <span>{progress.watchdog_summary}</span> : null}
          </Row>
        ) : null}
      </KeyValues>
      <p className="card-note" data-live-preview="" hidden={!preview}>
        {preview}
      </p>
    </Card>
  );
}

const RETRY_LABELS: Record<string, string> = {
  started: RETRY_MESSAGE,
  recovered: "重试成功，调用已恢复",
  failed: "本次自动重试失败",
};

function TaskBody({ workbench, state, projection, task }: { workbench: Workbench; state: WorkbenchState; projection: RunProjection; task: DagNode }) {
  if (state.activeTab === "artifacts") {
    return <RecordedArtifactCards artifacts={task.artifacts || {}} manifest={task.manifest_md} />;
  }
  if (state.activeTab === "events") {
    const retries = (task.provider_retry_history || []).slice().reverse();
    const events = (projection.events || []).slice().reverse();
    return (
      <>
        {retries.length ? (
          <Card title="调用重试记录">
            <KeyValues>
              {retries.map((event) => (
                <Row label={`#${event.event_seq} · ${event.actor}`} key={event.event_seq}>
                  {RETRY_LABELS[event.phase]}
                </Row>
              ))}
            </KeyValues>
          </Card>
        ) : null}
        <div className="event-log scroller">
          {events.length ? <EventRows events={events} /> : <Notice level="info">暂无事件</Notice>}
        </div>
      </>
    );
  }
  const checks = (task.acceptance_checks || [])
    .map((check) => `${check.passed ? "✓" : "✕"} ${check.id} · ${check.description}`)
    .join("\n");
  return (
    <>
      {task.provider_retry ? <Notice level="warn">{`${RETRY_MESSAGE}。下一步：${RETRY_NEXT_STEP}。`}</Notice> : null}
      {task.blocker ? (
        <Notice level="bad">
          {`受影响任务：${task.blocker.affected_task_ids.join("、")}。原因：${task.blocker.detail}`}
        </Notice>
      ) : null}
      {(task.blocked_by || []).map((taskId) => {
        const blocker = projection.dag.nodes.find((node) => node.task_id === taskId)?.blocker;
        if (!blocker) return null;
        return (
          <Notice
            level="bad"
            key={taskId}
            action={
              <button
                type="button"
                className="btn"
                data-blocker-task-id={taskId}
                onClick={() => workbench.showBlockerTask(taskId)}
              >
                {`查看 ${taskId} 的处理操作`}
              </button>
            }
          >
            {`依赖任务 ${taskId} 受阻：${blocker.detail}`}
          </Notice>
        );
      })}
      <LiveProgressCard state={state} task={task} />
      <Card title="执行概览">
        <KeyValues>
          <Row label="任务目标">{task.objective}</Row>
          <Row label="执行者" mono>
            {task.executor}
          </Row>
          <Row label="角色 / 类型" mono>
            {`${task.role || "engineer"} / ${task.kind || "implement"}`}
          </Row>
          <Row label="交付物">{(task.deliverables || []).join(", ") || "（未声明）"}</Row>
          <Row label="前置任务" mono>
            {(task.depends_on || []).join(", ") || "（无）"}
          </Row>
          <Row label="写入范围" mono>
            {(task.write_scope || []).join(", ") || "（只读）"}
          </Row>
          <Row label="诊断" mono>
            {`${task.diagnostic_tool} → ${task.diagnostic_output}`}
          </Row>
          <Row label="归因">{task.diagnostic_attribution}</Row>
        </KeyValues>
      </Card>
      {task.accepted_risk_reason ? <Notice level="warn">{`未验证 · 人工接受：${task.accepted_risk_reason}`}</Notice> : null}
      <Card title="验收检查">
        <pre className="code-block scroller">{checks || "（暂无验收检查）"}</pre>
      </Card>
    </>
  );
}

function taskDock(workbench: Workbench, state: WorkbenchState, projection: RunProjection, task: DagNode): DockContent {
  const meta = statusMeta(taskDisplayStatus(task, projection), task.accepted_risk_reason);
  const live = task.status === "running" || task.status === "verifying";
  const stream = task.subagent_stream || [];
  return {
    title: `${task.task_id} · ${task.title}`,
    stages: (
      <>
        <Badge variant={meta.variant}>{`${meta.glyph} ${meta.label}`}</Badge>
        <Badge>{`尝试 ${task.attempt} / ${task.max_cycles}`}</Badge>
        <Badge>{`审查 ${task.review_cycle} 轮`}</Badge>
        <Badge>{task.executor}</Badge>
        <Badge>{`角色 ${task.role || "engineer"}`}</Badge>
        <Badge>{`类型 ${task.kind || "implement"}`}</Badge>
      </>
    ),
    subpane: <SubPane title={`${task.task_id} · 沙盒终端`} tools={stream} tailLines={task.terminal_lines} live={live} />,
    tools: stream.length ? (
      stream.map((entry, index) => <ToolRow tool={entry} key={index} />)
    ) : (
      <Notice level="info">该任务尚未执行工具调用</Notice>
    ),
    tabs: <Tabs workbench={workbench} active={state.activeTab} />,
    body: <TaskBody workbench={workbench} state={state} projection={projection} task={task} />,
    receipt: <ReceiptBlock receipt={task.receipt} />,
  };
}

function workflowDock(projection: RunProjection, name: string): DockContent {
  const workflow = projection.workflow;
  const graph = workflow.graph || { nodes: [], edges: [] };
  const node = (graph.nodes || []).find((item) => item.name === name);
  const status = (workflow.node_statuses || {})[name] || "pending";
  const meta = statusMeta(status);
  const outgoing = (graph.edges || []).filter((edge) => edge.source === name);
  const incoming = (graph.edges || []).filter((edge) => edge.target === name);
  const events = (projection.events || [])
    .filter(
      (event) =>
        (event.payload && event.payload.node === name) ||
        event.stage === name ||
        (name === "intake" && event.event_type === "intake.triaged") ||
        (name === workflow.failed_node && event.event_type === "graph.failed"),
    )
    .slice(-16)
    .reverse();
  return {
    title: `${NODE_LABELS[name] || name} · ${name}`,
    stages: (
      <>
        <Badge variant={meta.variant}>{`${meta.glyph} ${meta.label}`}</Badge>
        <Badge>{`类型 ${node?.kind || "node"}`}</Badge>
        {node?.routed ? <Badge variant="warn">路由分叉</Badge> : <Badge>直连节点</Badge>}
        <Badge>{`最大访问 ${node?.max_visits || 1}`}</Badge>
        {node?.on_error ? <Badge variant="bad">{`失败 → ${node.on_error}`}</Badge> : null}
      </>
    ),
    body: (
      <>
        {workflow.failed_node === name && workflow.last_error ? (
          <Notice level="bad">{localizeError(workflow.last_error)}</Notice>
        ) : null}
        <Card title="节点定义">
          <KeyValues>
            <Row label="节点状态" mono>
              {status}
            </Row>
            <Row label="入边" mono>
              {incoming.map((edge) => edge.source).join(", ") || "（起始节点）"}
            </Row>
            <Row label="出边" mono>
              {outgoing.map((edge) => `${edge.target}${edge.conditional ? "（条件）" : ""}`).join(", ") || "（终点）"}
            </Row>
          </KeyValues>
        </Card>
        <Card title="节点事件">
          <div className="event-log scroller">
            {events.length ? <EventRows events={events} /> : <p className="view-note">该节点暂无事件</p>}
          </div>
        </Card>
      </>
    ),
  };
}

function deliveryDock(projection: RunProjection, review: DeliveryAcceptance, row: DeliveryCriterion): DockContent {
  const evidence = row.evidence!;
  const logUrl = deliveryLogUrl(projection.run_id, evidence.event_seq, evidence.check_position);
  const [label, variant] = DELIVERY_STATUS[row.status] || DELIVERY_STATUS.unverified;
  return {
    title: `${row.requirement_id} · 条件 ${row.criterion_index}`,
    stages: <Badge variant={variant}>{label}</Badge>,
    body: (
      <>
        <Card title="证据引用">
          <KeyValues>
            <Row label="验收条件">{row.text}</Row>
            <Row label={`GoalSpec v${review.spec_version}`} mono>
              {review.spec_hash}
            </Row>
            <Row label="交付版本" mono>
              {evidence.delivery_version}
            </Row>
            <Row label="执行记录">{`#${evidence.event_seq} · 退出码 ${evidence.exit_code}`}</Row>
          </KeyValues>
        </Card>
        <Card title="执行日志">
          <pre className="code-block scroller">{evidence.log}</pre>{" "}
          <a className="btn btn-quiet" href={logUrl} target="_blank" rel="noopener">
            打开完整日志
          </a>
        </Card>
      </>
    ),
  };
}

function dockContent(workbench: Workbench, state: WorkbenchState): DockContent {
  const projection = state.projection;
  if (!projection) {
    return {
      title: "节点执行详情",
      stages: null,
      tabs: <Tabs workbench={workbench} active={state.activeTab} />,
      body: null,
    };
  }
  const review = projection.delivery_acceptance;
  const selection = state.deliveryEvidence;
  const row = selection && review ? review.criteria[selection.index] : undefined;
  if (selection && review && row?.evidence) return deliveryDock(projection, review, row);
  if (state.workflowNode) return workflowDock(projection, state.workflowNode);
  const task = workbench.currentTask();
  if (task) return taskDock(workbench, state, projection, task);
  const workflow = projection.workflow;
  const focusNode = workflow.failed_node || workflow.current_node || workflow.graph?.nodes?.[0]?.name;
  if (focusNode) return workflowDock(projection, focusNode);
  return {
    title: "节点执行详情",
    stages: null,
    body: <p className="view-note">点击工作流或任务节点以查看事件与产物</p>,
  };
}

export function InspectorDock({ workbench, state }: { workbench: Workbench; state: WorkbenchState }) {
  const content = dockContent(workbench, state);
  return (
    <aside id="dock" className="scroller" aria-label="任务检查器">
      <header className="dock-head">
        <div>
          <span className="dock-kicker">检查器</span>
          <h2 className="dock-title" id="dock-title" tabIndex={-1}>
            {content.title}
          </h2>
        </div>
        <button type="button" className="btn btn-quiet" id="btn-dock-close" aria-label="收起检查器" onClick={() => workbench.closeDock()}>
          ⇥
        </button>
      </header>
      <div className="dock-stages" id="dock-stages">
        {content.stages}
      </div>
      <div id="dock-subpane">{content.subpane}</div>
      <div className="tool-list" id="dock-tools">
        {content.tools}
      </div>
      <div id="dock-tabs">{content.tabs}</div>
      <div className="dock-body" id="dock-body">
        {content.body}
      </div>
      <div id="dock-receipt">{content.receipt}</div>
    </aside>
  );
}
