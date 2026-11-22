import { flushSync } from "react-dom";
import { useSyncExternalStore } from "react";

import {
  createRun,
  fetchAttemptArtifacts,
  fetchWorkbenchOptions,
  listRuns,
  postCommand,
  seedDemo,
  type WorkbenchOptions,
} from "../api/client";
import type { StreamPhase } from "../api/useRunStream";
import type {
  ActivityPreviews,
  ApprovalAction,
  CommandBody,
  CreateRunBody,
  DagNode,
  GoalSpec,
  KeyOutputItem,
  RecordedAttemptArtifacts,
  ReviewArtifact,
  RunProjection,
  RunSummary,
} from "../api/types";
import {
  APPROVAL_LABELS,
  currentStage,
  isTab,
  isView,
  pendingAmbiguities,
  taskBlocker,
  unresolvedAmbiguities,
  type Tab,
  type View,
} from "../lib/labels";
import { captureReadingPosition, highlightProgress } from "../lib/readingPosition";

export type Theme = "light" | "dark";

export interface DeliverySelection {
  index: number;
  specHash: string;
  runId: string;
}

export interface ApprovalCommand {
  action: ApprovalAction;
  target_hash: string;
  expected_state_revision: number;
}

/** A review card bound to the exact artifact fingerprint the human is looking at. */
export interface ApprovalBinding {
  artifact: ReviewArtifact;
  baseline: ReviewArtifact | null;
  reviewKey: string;
  command: ApprovalCommand;
}

export interface ApprovalSnapshot {
  runId: string;
  revision: number;
  reviews: Map<ApprovalAction, ApprovalBinding>;
}

export type OriginVersion = "candidate" | "baseline";

export interface ApprovalOrigin {
  version: OriginVersion;
  ref: string;
  text: string;
  source: string | null;
}

export interface DialogConfig {
  title: string;
  description: string;
  label: string;
  defaultValue: string;
  placeholder?: string;
  requireNonEmpty: boolean;
  onConfirm: (text: string) => Promise<void>;
}

export type AttemptDetail =
  | { status: "loading" }
  | { status: "loaded"; artifacts: RecordedAttemptArtifacts };

export interface NewRunForm {
  demoId: string;
  title: string;
  workflow: string;
  provider: string;
  model: string;
  workerDriver: string;
  reviewerDriver: string;
  monitorModel: string;
  input: string;
}

export interface WorkbenchState {
  runs: RunSummary[];
  runId: string | null;
  projection: RunProjection | null;
  view: View;
  selectedTaskId: string | null;
  workflowNode: string | null;
  deliveryEvidence: DeliverySelection | null;
  activeTab: Tab;
  approvalSnapshot: ApprovalSnapshot | null;
  approvalOrigins: Partial<Record<ApprovalAction, ApprovalOrigin>>;
  openedPlanGraphHash: string | null;
  options: WorkbenchOptions | null;
  clarifyGeneration: number;
  userOpenedDock: boolean;
  dockOpen: boolean;
  railCollapsed: boolean | null;
  theme: Theme;
  submitting: boolean;
  phase: StreamPhase | null;
  activity: ActivityPreviews | undefined;
  streamGeneration: number;
  attemptDetails: Record<string, AttemptDetail>;
  toast: { message: string; visible: boolean };
  dialog: DialogConfig | null;
}

const PHASE_TEXT: Partial<Record<StreamPhase, string>> = {
  loading: "正在加载快照…",
  missing: "找不到运行",
  reconnecting: "正在重连 · 保留上次确认",
};

const byId = <T extends HTMLElement = HTMLElement>(id: string): T | null =>
  document.getElementById(id) as T | null;

/** Resolve an RFC 6901 JSON pointer inside an artifact. */
function resolvePointer(artifact: unknown, ref: string): { parts: string[]; value: unknown } {
  const parts = ref
    .slice(1)
    .split("/")
    .map((key) => key.replaceAll("~1", "/").replaceAll("~0", "~"));
  const value = parts.reduce<unknown>(
    (item, key) => (item && typeof item === "object" ? (item as Record<string, unknown>)[key] : undefined),
    artifact,
  );
  return { parts, value };
}

export class Workbench {
  readonly params: URLSearchParams;
  readonly sse: boolean;
  state: WorkbenchState;
  private version = 0;
  private listeners = new Set<() => void>();
  private toastTimer: number | undefined;
  private pendingAction: DialogConfig | null = null;

  constructor(location: Location = window.location) {
    this.params = new URLSearchParams(location.search);
    const hashView = location.hash.replace(/^#/, "");
    const queryView = this.params.get("view");
    const queryTab = this.params.get("tab");
    this.sse = this.params.get("nosse") !== "1";
    this.state = {
      runs: [],
      runId: null,
      projection: null,
      view: isView(hashView) ? hashView : isView(queryView) ? queryView : "topology",
      selectedTaskId: this.params.get("task") || null,
      workflowNode: null,
      deliveryEvidence: null,
      activeTab: isTab(queryTab) ? queryTab : "overview",
      approvalSnapshot: null,
      approvalOrigins: {},
      openedPlanGraphHash: null,
      options: null,
      clarifyGeneration: 0,
      userOpenedDock: false,
      dockOpen: false,
      railCollapsed: null,
      theme: document.documentElement.dataset.theme === "dark" ? "dark" : "light",
      submitting: false,
      phase: null,
      activity: undefined,
      streamGeneration: 0,
      attemptDetails: {},
      toast: { message: "", visible: false },
      dialog: null,
    };
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getVersion = (): number => this.version;

  /** Publish the mutated state; `sync` commits to the DOM before returning. */
  private commit(sync = true): void {
    this.version += 1;
    const notify = () => this.listeners.forEach((listener) => listener());
    if (sync) flushSync(notify);
    else notify();
  }

  async start(): Promise<void> {
    window.addEventListener("hashchange", () => {
      const view = window.location.hash.replace(/^#/, "");
      if (isView(view) && view !== this.state.view) this.setView(view);
    });
    this.setView(this.state.view);
    this.setTheme(this.state.theme);
    let data = await this.refreshRuns();
    if (!this.state.runs.length) {
      await seedDemo();
      data = await this.refreshRuns();
    }
    const initialRun =
      this.params.get("run") || data.default_run_id || this.state.runs[0]?.run_id || null;
    if (initialRun) this.loadRun(initialRun);
  }

  async refreshRuns() {
    const data = await listRuns();
    this.state.runs = data.runs || [];
    this.commit();
    return data;
  }

  /** Switch the console to another run; the transport hook reconnects on `streamGeneration`. */
  loadRun(runId: string): void {
    const s = this.state;
    if (s.runId !== runId) {
      this.pendingAction = null;
      s.dialog = null;
      const hitlDialog = byId<HTMLDialogElement>("hitl-dialog");
      if (hitlDialog?.open) hitlDialog.close();
    }
    s.runId = runId;
    s.userOpenedDock = false;
    s.activity = undefined;
    s.attemptDetails = {};
    s.streamGeneration += 1;
    this.syncUrl();
    this.setPhase("loading", true);
  }

  selectRun(runId: string): void {
    if (!runId || runId === this.state.runId) return;
    this.state.selectedTaskId = null;
    this.state.workflowNode = null;
    this.loadRun(runId);
  }

  connectionText(): string {
    const s = this.state;
    return s.submitting ? "正在提交…" : (s.phase && PHASE_TEXT[s.phase]) || "";
  }

  setPhase(phase: StreamPhase, force = false): void {
    const before = this.connectionText();
    this.state.phase = phase;
    if (!force && this.connectionText() === before) return;
    const restore = this.state.projection ? captureReadingPosition() : null;
    this.commit();
    restore?.();
  }

  setActivity(previews: ActivityPreviews | undefined): void {
    this.state.activity = previews;
    this.commit(false);
  }

  /** Apply a projection unless it is older than (or identical to) the one on screen. */
  applyProjection = (projection: RunProjection): void => {
    const s = this.state;
    if (projection.run_id !== s.runId) return;
    const switchedRun = s.projection?.run_id !== projection.run_id;
    const previous = switchedRun ? null : s.projection;
    if (
      previous &&
      (projection.state_revision < previous.state_revision ||
        projection.latest_event_seq < previous.latest_event_seq ||
        (projection.state_revision === previous.state_revision &&
          projection.latest_event_seq === previous.latest_event_seq))
    ) {
      return;
    }
    const restore = switchedRun ? null : captureReadingPosition();
    if (previous?.plan?.content_hash !== projection.plan?.content_hash) {
      s.openedPlanGraphHash = null;
    }
    s.projection = projection;
    const nodes = projection.dag.nodes || [];
    if (!s.selectedTaskId || !nodes.some((node) => node.task_id === s.selectedTaskId)) {
      const blocked = nodes.find((node) => node.status === "blocked" || node.status === "failed");
      s.selectedTaskId = (blocked || nodes[0])?.task_id || null;
    }
    const runEntry = s.runs.find((run) => run.run_id === projection.run_id);
    if (runEntry) {
      runEntry.status = projection.status;
      runEntry.workflow_name = projection.workflow_name || runEntry.workflow_name;
    }
    if (switchedRun && !s.userOpenedDock) {
      s.dockOpen = projection.hitl_banner?.kind === "task_blocked";
    }
    this.syncApprovals(projection);
    this.normalizeDock();
    this.commit();
    this.afterProjection(projection, previous);
    restore?.();
  };

  private afterProjection(projection: RunProjection, previous: RunProjection | null): void {
    const approvedPlan = projection.plan_approved ? projection.plan : null;
    if (
      !approvedPlan ||
      !previous?.plan_approved ||
      approvedPlan.content_hash !== previous.plan?.content_hash
    ) {
      const detail = byId<HTMLDetailsElement>("run-plan-detail");
      if (detail) detail.open = false;
    }
    if (!previous) return;
    if (currentStage(previous).key !== currentStage(projection).key) {
      highlightProgress(byId("run-current-stage"));
    }
    const latest = projection.key_outputs?.[0];
    if (latest && previous.key_outputs?.[0]?.key !== latest.key) {
      highlightProgress(byId("run-key-output"));
    }
  }

  private syncApprovals(projection: RunProjection): void {
    const s = this.state;
    const previous = s.approvalSnapshot;
    if (previous?.runId === projection.run_id && previous.revision > projection.state_revision) return;
    const reviews = new Map<ApprovalAction, ApprovalBinding>();
    for (const [action, review] of Object.entries(projection.approval_reviews || {})) {
      if (!(action in APPROVAL_LABELS)) continue;
      const artifact = projection[review.artifact] as ReviewArtifact | null | undefined;
      if (artifact?.content_hash !== review.target_hash) continue;
      reviews.set(action as ApprovalAction, {
        artifact,
        baseline: review.baseline ?? null,
        reviewKey: JSON.stringify([
          review.target_hash,
          review.baseline?.content_hash,
          review.comparison_gap,
          review.sections,
          review.unchanged_sections,
        ]),
        command: {
          action: action as ApprovalAction,
          target_hash: review.target_hash,
          expected_state_revision: projection.state_revision,
        },
      });
    }
    const sameRun = previous?.runId === projection.run_id;
    const origins: Partial<Record<ApprovalAction, ApprovalOrigin>> = {};
    for (const [action, binding] of reviews) {
      const origin = s.approvalOrigins[action];
      if (origin && sameRun && previous?.reviews.get(action)?.reviewKey === binding.reviewKey) {
        origins[action] = origin;
      }
    }
    s.approvalOrigins = origins;
    s.approvalSnapshot = { runId: projection.run_id, revision: projection.state_revision, reviews };
  }

  /** Drop stale inspector selections and pin the task the dock is showing. */
  private normalizeDock(): void {
    const s = this.state;
    const projection = s.projection;
    if (!projection) return;
    const review = projection.delivery_acceptance;
    const selection = s.deliveryEvidence;
    if (
      !(
        selection &&
        selection.runId === projection.run_id &&
        selection.specHash === review?.spec_hash &&
        review.criteria[selection.index]?.evidence
      )
    ) {
      s.deliveryEvidence = null;
    }
    if (!s.deliveryEvidence && !s.workflowNode) {
      const task = this.currentTask();
      if (task) s.selectedTaskId = task.task_id;
    }
  }

  currentTask(): DagNode | null {
    const nodes = this.state.projection?.dag.nodes || [];
    return nodes.find((node) => node.task_id === this.state.selectedTaskId) || nodes[0] || null;
  }

  private syncUrl(): void {
    const url = new URL(window.location.href);
    if (this.state.runId) url.searchParams.set("run", this.state.runId);
    url.hash = this.state.view;
    if (url.href !== window.location.href) window.history.replaceState(window.history.state, "", url);
  }

  setView(view: View): void {
    this.state.view = view;
    this.syncUrl();
    this.commit();
  }

  setTheme(theme: Theme): void {
    this.state.theme = theme;
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("olympus-theme", theme);
    } catch {
      // Storage may be unavailable (private mode); the theme still applies for this page.
    }
    this.commit();
  }

  toggleTheme(): void {
    this.setTheme(this.state.theme === "dark" ? "light" : "dark");
  }

  toggleRail(): void {
    this.state.railCollapsed = this.state.railCollapsed !== true;
    this.commit();
  }

  setDockOpen(open: boolean, byUser: boolean): void {
    this.state.userOpenedDock = byUser && open;
    this.state.dockOpen = open;
    this.commit();
  }

  closeDock(): void {
    this.setDockOpen(false, false);
    const selection = this.state.deliveryEvidence;
    if (selection) {
      document
        .querySelector<HTMLElement>(`#delivery-criteria [data-delivery-evidence="${selection.index}"]`)
        ?.focus();
    }
  }

  toast(message: string): void {
    this.state.toast = { message, visible: true };
    this.commit(false);
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => {
      this.state.toast = { ...this.state.toast, visible: false };
      this.commit(false);
    }, 3200);
  }

  selectTask = (taskId: string): void => {
    const s = this.state;
    if (!s.projection || !taskId) return;
    s.deliveryEvidence = null;
    s.workflowNode = null;
    s.selectedTaskId = taskId;
    s.userOpenedDock = true;
    s.dockOpen = true;
    this.normalizeDock();
    this.commit();
  };

  selectWorkflowNode = (name: string): void => {
    const s = this.state;
    if (!s.projection || !name) return;
    s.deliveryEvidence = null;
    s.workflowNode = name;
    s.userOpenedDock = true;
    s.dockOpen = true;
    this.commit();
  };

  setTab(tab: Tab): void {
    this.state.activeTab = tab;
    this.commit();
  }

  showDeliveryEvidence(index: number): void {
    const s = this.state;
    const review = s.projection?.delivery_acceptance;
    if (!s.projection || !review) return;
    s.deliveryEvidence = { index, specHash: review.spec_hash, runId: s.projection.run_id };
    s.userOpenedDock = true;
    s.dockOpen = true;
    this.normalizeDock();
    this.commit();
    byId("dock-title")?.focus();
  }

  showBlockerTask(taskId: string): void {
    this.selectTask(taskId);
    byId("btn-banner-retry")?.focus();
  }

  bannerTaskId(): string | null {
    return taskBlocker(this.state.projection, this.state.selectedTaskId)?.task_id || null;
  }

  inspectBannerTask(): void {
    const taskId = this.bannerTaskId();
    if (!taskId) return;
    this.state.activeTab = "overview";
    this.selectTask(taskId);
  }

  inspectFailure(): void {
    const workflow = this.state.projection?.workflow;
    this.selectWorkflowNode(workflow?.failed_node || workflow?.current_node || "align");
  }

  openTaskGraph(planHash: string | null | undefined): void {
    const s = this.state;
    if (!planHash || s.projection?.plan?.content_hash !== planHash) return;
    s.openedPlanGraphHash = planHash;
    this.setView("topology");
    byId("topology-title")?.focus();
  }

  async openOutput(output: KeyOutputItem | undefined): Promise<void> {
    const s = this.state;
    if (!output || !s.projection) return;
    if (output.current && (output.kind === "spec" || output.kind === "design")) {
      this.setView("design");
      const detail = byId(`run-${output.kind}-detail`);
      if (detail) {
        detail.tabIndex = -1;
        detail.focus();
      }
      return;
    }
    if (output.kind === "plan" && output.current) {
      this.openTaskGraph(s.projection.plan?.content_hash);
      return;
    }
    if (output.current) {
      s.activeTab = "artifacts";
      this.selectTask(output.artifact_id);
      byId("dock-title")?.focus();
      return;
    }
    const history = byId<HTMLDetailsElement>("run-output-history");
    if (!history) return;
    history.open = true;
    if (!output.detail_available) {
      history.querySelector<HTMLElement>("summary")?.focus();
      return;
    }
    const rowSelector = `[data-output-key="${CSS.escape(output.key)}"]`;
    if (!history.querySelector(rowSelector)) return;
    const focusDetail = () =>
      history.querySelector<HTMLElement>(`${rowSelector} .run-attempt-detail`)?.focus();
    if (s.attemptDetails[output.key]) {
      focusDetail();
      return;
    }
    s.attemptDetails = { ...s.attemptDetails, [output.key]: { status: "loading" } };
    this.commit();
    focusDetail();
    const runId = s.runId;
    try {
      if (!runId || output.version == null) throw new Error("missing attempt");
      const artifacts = await fetchAttemptArtifacts(runId, output.artifact_id, output.version);
      if (this.state.runId !== runId) return;
      s.attemptDetails = { ...s.attemptDetails, [output.key]: { status: "loaded", artifacts } };
      this.commit(false);
    } catch {
      if (this.state.runId !== runId) return;
      const { [output.key]: _dropped, ...rest } = s.attemptDetails;
      s.attemptDetails = rest;
      this.commit(false);
      this.toast("读取执行产物失败，可重新打开");
    }
  }

  showApprovalOrigin(action: ApprovalAction, ref: string, baseline: boolean): void {
    const binding = this.reviewBinding(action);
    const artifact = baseline ? binding?.baseline : binding?.artifact;
    if (!binding || !artifact) return;
    const { parts, value } = resolvePointer(artifact, ref);
    if (value === undefined) return;
    const requirements = (artifact as { requirements?: { source_ref?: string | null }[] }).requirements;
    const source = parts[0] === "requirements" ? requirements?.[Number(parts[1])]?.source_ref ?? null : null;
    const version: OriginVersion = baseline ? "baseline" : "candidate";
    this.state.approvalOrigins = {
      ...this.state.approvalOrigins,
      [action]: { version, ref, text: typeof value === "string" ? value : JSON.stringify(value), source },
    };
    this.commit();
    const card = document.querySelector(`[data-review-action="${action}"]`);
    const details = card?.querySelector<HTMLDetailsElement>(`[data-review-disclosure="${version}"]`);
    if (!details) return;
    details.open = true;
    details.querySelector<HTMLElement>(".approval-origin code")?.focus();
  }

  private reviewBinding(action: ApprovalAction): ApprovalBinding | undefined {
    const snapshot = this.state.approvalSnapshot;
    if (snapshot?.runId !== this.state.runId) return undefined;
    return snapshot?.reviews.get(action);
  }

  async approve(action: ApprovalAction): Promise<void> {
    const snapshot = this.state.approvalSnapshot;
    const binding = this.reviewBinding(action);
    if (!snapshot || !binding) return;
    const { artifact, command } = binding;
    const { runId } = snapshot;
    if (command.action === "approve_spec") {
      const spec = artifact as GoalSpec;
      const blockers = unresolvedAmbiguities(spec);
      const blockingCritique = (spec.minority_notes || []).some(
        (note) => note.severity === "blocking" && !note.resolved,
      );
      if (blockers.length || blockingCritique) {
        this.openDialog({
          title: `GoalSpec v${spec.version} 仍有未决阻断项`,
          description: "请先处理阻断项。若要批准此版本，请填写接受风险的理由。",
          label: "接受风险理由（必填）",
          defaultValue: "首期范围已与业务确认，接受未决歧义风险先行冻结规格。",
          requireNonEmpty: true,
          onConfirm: (reason) =>
            this.sendCommand(
              { ...command, action: "approve_spec", accept_risk: true, reason },
              "已带风险批准 GoalSpec，进入架构设计",
              runId,
            ),
        });
        return;
      }
      await this.sendCommand(command, "已批准 GoalSpec，进入架构设计", runId);
    } else if (command.action === "approve_design") {
      await this.sendCommand(command, "已批准 DesignSpec，启动任务分解", runId);
    } else {
      const message =
        this.state.projection?.workflow?.stop_after === "plan"
          ? "已批准任务计划，蓝图编排已完成"
          : "已批准任务计划，开始执行";
      await this.sendCommand(command, message, runId);
    }
  }

  /** Post a command bound to the revision on screen; a 409 applies the fresh projection. */
  async sendCommand(body: CommandBody, successMessage: string, runId = this.state.runId): Promise<void> {
    const s = this.state;
    if (!runId || runId !== s.runId || s.projection?.run_id !== runId) return;
    const payload: CommandBody = {
      operation_id: `op-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      expected_state_revision: s.projection.state_revision,
      actor: "zeus",
      ...body,
    };
    let outcome;
    try {
      outcome = await postCommand(runId, payload);
    } catch (error) {
      if (this.state.runId === runId) this.toast(`操作请求失败：${(error as Error).message}`);
      return;
    }
    if (this.state.runId !== runId) return;
    if (outcome.kind === "conflict") {
      const fresh = outcome.projection;
      if (fresh) {
        this.applyProjection(fresh);
        this.refreshRuns().catch(() => undefined);
      }
      this.toast(`状态已同步至最新修订 (r${fresh ? fresh.state_revision : "?"})，请确认后重试`);
      return;
    }
    if (outcome.kind === "rejected") {
      this.toast(`操作被拒绝：${outcome.error}`);
      return;
    }
    this.applyProjection(outcome.projection);
    this.refreshRuns().catch(() => undefined);
    this.toast(successMessage);
  }

  togglePause(): Promise<void> {
    const action = this.state.projection?.status === "paused" ? "resume" : "pause";
    return this.sendCommand({ action }, action === "pause" ? "已暂停新任务派发" : "已恢复任务编排调度");
  }

  promoteToExecute(): Promise<void> {
    return this.sendCommand(
      { action: "promote_to_execute", workflow_name: "full_sdlc_tdd" },
      "已升级为 full_sdlc_tdd，启动沙盒代码执行",
    );
  }

  retryWorkflow(): Promise<void> {
    return this.sendCommand({ action: "retry_workflow" }, "已从断点重新启动工作流");
  }

  async retryBannerTask(): Promise<void> {
    const projection = this.state.projection;
    const taskId = this.bannerTaskId();
    if (!taskId || !projection || projection.run_id !== this.state.runId) return;
    await this.sendCommand(
      {
        action: "retry_task",
        task_id: taskId,
        hint: "",
        expected_state_revision: projection.state_revision,
      },
      `已将任务 ${taskId} 重置为就绪 (READY)`,
      projection.run_id,
    );
  }

  hintBannerTask(): void {
    const projection = this.state.projection;
    const taskId = this.bannerTaskId();
    if (!taskId || !projection || projection.run_id !== this.state.runId) return;
    const targetRunId = projection.run_id;
    const expectedRevision = projection.state_revision;
    this.openDialog({
      title: `向任务 ${taskId} 注入人工提示并重试`,
      description:
        "提示原文保留在审计事件中，随下次隔离沙盒执行一并注入 Worker 上下文，不直接篡改需求契约。",
      label: "注入提示内容",
      defaultValue: "",
      placeholder: "输入要注入的人工提示，例如需修复的边界或重试思路。",
      requireNonEmpty: true,
      onConfirm: (hint) =>
        this.sendCommand(
          {
            action: "retry_task",
            task_id: taskId,
            hint,
            expected_state_revision: expectedRevision,
          },
          `已向 ${taskId} 注入提示并重新排入就绪队列`,
          targetRunId,
        ),
    });
  }

  reassignBannerTask(): void {
    const projection = this.state.projection;
    const taskId = this.bannerTaskId();
    if (!taskId || !projection || projection.run_id !== this.state.runId) return;
    const targetRunId = projection.run_id;
    const expectedRevision = projection.state_revision;
    this.openDialog({
      title: `将任务 ${taskId} 改派给人工工程师`,
      description:
        "改派后人工提交的产物仍须经过相同的程序门禁与 Argus 独立复核才能成为已验收状态。",
      label: "改派备注",
      defaultValue: "",
      placeholder: "输入改派备注，例如接手人与需关注的验收要点。",
      requireNonEmpty: true,
      onConfirm: (note) =>
        this.sendCommand(
          {
            action: "reassign_human",
            task_id: taskId,
            note,
            expected_state_revision: expectedRevision,
          },
          `已记录 ${taskId} 改派人工并回到就绪态`,
          targetRunId,
        ),
    });
  }

  acceptBannerTaskRisk(): void {
    const projection = this.state.projection;
    const taskId = this.bannerTaskId();
    if (!taskId || !projection || projection.run_id !== this.state.runId) return;
    const targetRunId = projection.run_id;
    const expectedRevision = projection.state_revision;
    this.openDialog({
      title: `接受风险完成任务 ${taskId}`,
      description:
        "必须填写明确理由。该决定与 Agent 建议分开存储，追踪矩阵标记为“未验证 · 人工接受”，绝不显示为“已验证”。",
      label: "接受风险理由（必填）",
      defaultValue: "",
      placeholder: "输入接受风险的明确理由，例如影响范围与缓解措施。",
      requireNonEmpty: true,
      onConfirm: (reason) =>
        this.sendCommand(
          {
            action: "accept_risk",
            task_id: taskId,
            reason,
            expected_state_revision: expectedRevision,
          },
          `已记录人工接受风险，${taskId} 标记为未验证 · 人工接受`,
          targetRunId,
        ),
    });
  }

  async submitClarify(answers: Record<string, string>): Promise<void> {
    const projection = this.state.projection;
    if (!projection) return;
    const missing = pendingAmbiguities(projection).filter((amb) => amb.blocking && !answers[amb.id]);
    if (missing.length) {
      this.toast(`还有 ${missing.length} 个阻断歧义未回答`);
      return;
    }
    if (!Object.keys(answers).length) {
      this.toast("请先回答至少一个澄清问题");
      return;
    }
    this.state.clarifyGeneration += 1;
    await this.sendCommand({ action: "clarify", answers }, "已提交澄清并生成新的 GoalSpec 版本");
  }

  openDialog(config: DialogConfig): void {
    this.pendingAction = config;
    this.state.dialog = config;
    this.commit();
    const input = byId<HTMLTextAreaElement>("hitl-dialog-input");
    if (input) {
      input.value = config.defaultValue || "";
      input.placeholder = config.placeholder || "";
    }
    byId<HTMLDialogElement>("hitl-dialog")?.showModal();
  }

  /** `close` of #hitl-dialog: only an explicit confirm submits; Esc / light dismiss cancel. */
  async closeDialog(returnValue: string): Promise<void> {
    const pending = this.pendingAction;
    this.pendingAction = null;
    if (this.state.dialog !== null) {
      this.state.dialog = null;
      this.commit();
    }
    if (returnValue !== "confirm" || !pending) return;
    const text = (byId<HTMLTextAreaElement>("hitl-dialog-input")?.value || "").trim();
    if (pending.requireNonEmpty && !text) {
      this.toast("必须填写理由或提示后才能提交");
      return;
    }
    await pending.onConfirm(text);
  }

  async openNewRun(): Promise<void> {
    this.state.workflowNode = null;
    byId<HTMLDialogElement>("new-run-dialog")?.showModal();
    try {
      await this.loadNewRunOptions();
    } catch {
      this.toast("加载工作流或提供方失败");
    }
  }

  private async loadNewRunOptions(): Promise<void> {
    if (this.state.options) return;
    this.state.options = await fetchWorkbenchOptions();
    this.commit(false);
  }

  async submitNewRun(form: NewRunForm): Promise<boolean> {
    const s = this.state;
    const rawInput = form.input.trim();
    if (!rawInput) {
      this.toast("请先填写需求原文或导入文件");
      return false;
    }
    const provider = form.provider || "auto";
    const model = form.model.trim();
    const providerConfig: NonNullable<CreateRunBody["provider_config"]> =
      provider === "auto" ? {} : { provider };
    if (model) providerConfig.model = model;
    const demoId = form.demoId.trim();
    const selectedDemo = demoId ? s.options?.demos.find((item) => item.id === demoId) : undefined;
    const payload: CreateRunBody = {
      raw_input: rawInput,
      title: form.title.trim() || rawInput.slice(0, 60),
      workflow_name: form.workflow || "product_blueprint",
      provider_config: providerConfig,
    };
    const overrides = [
      ["worker_driver", form.workerDriver],
      ["reviewer_driver", form.reviewerDriver],
      ["monitor_model", form.monitorModel],
    ] as const;
    for (const [name, raw] of overrides) {
      const value = raw.trim();
      if (value) payload[name] = value;
    }
    if (demoId) {
      payload.demo_id = demoId;
      if (selectedDemo && Array.isArray(selectedDemo.constraints)) {
        payload.constraints = selectedDemo.constraints;
      }
    }
    if (s.submitting) return false;
    s.submitting = true;
    this.renderConnection();
    let runId: string;
    try {
      runId = (await createRun(payload)).run_id;
    } catch (error) {
      this.toast(`创建请求失败：${(error as Error).message}`);
      return false;
    } finally {
      s.submitting = false;
      this.renderConnection();
    }
    s.selectedTaskId = null;
    s.workflowNode = null;
    this.toast(`已接收编排 ${runId}`);
    this.loadRun(runId);
    this.refreshRuns().catch(() => undefined);
    return true;
  }

  private renderConnection(): void {
    const restore = this.state.projection ? captureReadingPosition() : null;
    this.commit();
    restore?.();
  }
}

export function useWorkbench(workbench: Workbench): WorkbenchState {
  useSyncExternalStore(workbench.subscribe, workbench.getVersion);
  return workbench.state;
}
