import { flushSync } from "react-dom";
import { useSyncExternalStore } from "react";

import {
  cloneWorkspace,
  createRun,
  createWorkspace,
  deleteWorkspace,
  fetchAttemptArtifacts,
  fetchWorkbenchOptions,
  listRuns,
  listWorkspaces,
  postCommand,
  runExists,
  seedDemo,
  type WorkbenchOptions,
} from "../api/client";
import type { StreamPhase } from "../api/useRunStream";
import type {
  ActivityPreviews,
  ApprovalAction,
  CloneWorkspaceBody,
  CommandBody,
  CreateRunBody,
  CreateWorkspaceBody,
  DagNode,
  GoalSpec,
  KeyOutputItem,
  PantheonGod,
  PantheonSeatSummary,
  RecordedAttemptArtifacts,
  ReviewArtifact,
  RunListResponse,
  RunProjection,
  RunSummary,
  TaskType,
  WorkspaceMode,
  WorkspaceSummary,
} from "../api/types";
import { localized } from "../i18n/localized";
import { tr } from "../i18n/translate";
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
export type ActiveSurface = "board" | "oracle";
export type BoardModeFilter = "all" | WorkspaceMode;

const OPEN_TABS_KEY = "olympus-open-tabs";

function readStoredTabs(): string[] {
  try {
    const raw = localStorage.getItem(OPEN_TABS_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const seen = new Set<string>();
    const tabs: string[] = [];
    for (const item of parsed) {
      if (typeof item === "string" && item.trim() && !seen.has(item)) {
        seen.add(item);
        tabs.push(item);
      }
    }
    return tabs;
  } catch {
    return [];
  }
}

function isStorageUnavailable(error: unknown): boolean {
  return error instanceof DOMException || error instanceof ReferenceError;
}

function writeStoredTabs(tabs: string[]): boolean {
  try {
    localStorage.setItem(OPEN_TABS_KEY, JSON.stringify(tabs));
    return true;
  } catch (error) {
    if (!isStorageUnavailable(error)) throw error;
    return false;
  }
}

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
  workspaceMode: WorkspaceMode;
  workspaceId: string;
  customWorkspaceRoot: string;
  greenfieldPath: string;
  initGit: boolean;
  contextWorkspaceId: string;
  baseRef: string;
  scopePaths: string;
  taskType: TaskType;
  demoId: string;
  title: string;
  workflow: string;
  provider: string;
  model: string;
  workerDriver: string;
  reviewerDriver: string;
  input: string;
}

export type NewRunSubmitResult =
  | { ok: true; runId: string }
  | { ok: false; error: string };

export interface WorkbenchState {
  activeSurface: ActiveSurface;
  workspaces: WorkspaceSummary[];
  selectedWorkspaceId: string | null;
  boardModeFilter: BoardModeFilter;
  selectedGodFilter: PantheonGod | null;
  hoveredGod: PantheonGod | null;
  showArchived: boolean;
  openTabs: string[];
  quickDrawerRunId: string | null;
  newRunOpenSeq: number;
  workspaceDialogOpenSeq: number;
  allRuns: RunSummary[];
  runs: RunSummary[];
  pantheon: PantheonSeatSummary[];
  pendingRulingTotal: number;
  runId: string | null;
  projection: RunProjection | null;
  view: View;
  selectedTaskId: string | null;
  workflowNode: string | null;
  deliveryEvidence: DeliverySelection | null;
  productInspectionRunId: string | null;
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

/** Connection phase copy; proxied so `connectionText()` reads it in the active locale. */
const PHASE_TEXT: Partial<Record<StreamPhase, string>> = localized<Partial<Record<StreamPhase, string>>>({
  loading: "正在加载快照…",
  missing: "找不到运行",
  reconnecting: "正在重连 · 保留上次确认",
});

const byId = <T extends HTMLElement = HTMLElement>(id: string): T | null =>
  document.getElementById(id) as T | null;

/** Resolve an RFC 6901 JSON pointer inside an artifact. */
export function resolvePointer(artifact: unknown, ref: string): { parts: string[]; value: unknown } {
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

export function buildApprovalSnapshot(
  projection: RunProjection,
  previousSnapshot: ApprovalSnapshot | null,
  previousOrigins: Partial<Record<ApprovalAction, ApprovalOrigin>>,
): {
  snapshot: ApprovalSnapshot;
  origins: Partial<Record<ApprovalAction, ApprovalOrigin>>;
} | null {
  if (
    previousSnapshot?.runId === projection.run_id &&
    previousSnapshot.revision > projection.state_revision
  ) {
    return null;
  }
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
  const sameRun = previousSnapshot?.runId === projection.run_id;
  const origins: Partial<Record<ApprovalAction, ApprovalOrigin>> = {};
  for (const [action, binding] of reviews) {
    const origin = previousOrigins[action];
    if (origin && sameRun && previousSnapshot?.reviews.get(action)?.reviewKey === binding.reviewKey) {
      origins[action] = origin;
    }
  }
  return {
    snapshot: { runId: projection.run_id, revision: projection.state_revision, reviews },
    origins,
  };
}

export type RunCommandResult =
  | { kind: "ok"; projection: RunProjection }
  | { kind: "conflict"; projection: RunProjection | null; message: string }
  | { kind: "rejected"; message: string }
  | { kind: "network_error"; message: string };

/** Submit one revision-bound HITL command with unified operation_id and 409 conflict messaging. */
export async function executeRunCommand(
  runId: string,
  expectedRevision: number,
  body: CommandBody,
): Promise<RunCommandResult> {
  if (document.querySelector("#run-replay:modal")) {
    return { kind: "rejected", message: tr("只读回放") };
  }
  const payload: CommandBody = {
    operation_id: `op-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    expected_state_revision: expectedRevision,
    actor: "zeus",
    ...body,
  };
  let outcome;
  try {
    outcome = await postCommand(runId, payload);
  } catch (error) {
    return {
      kind: "network_error",
      message: tr("操作请求失败：{error}", { error: (error as Error).message }),
    };
  }
  if (outcome.kind === "conflict") {
    const fresh = outcome.projection;
    return {
      kind: "conflict",
      projection: fresh,
      message: tr("状态已同步至最新修订 (r{rev})，请确认后重试", {
        rev: fresh ? fresh.state_revision : "?",
      }),
    };
  }
  if (outcome.kind === "rejected") {
    return { kind: "rejected", message: tr("操作被拒绝：{error}", { error: outcome.error }) };
  }
  return { kind: "ok", projection: outcome.projection };
}

export class Workbench {
  readonly params: URLSearchParams;
  readonly sse: boolean;
  state: WorkbenchState;
  private version = 0;
  private listeners = new Set<() => void>();
  private toastTimer: number | undefined;
  private pendingAction: DialogConfig | null = null;
  private lastBoardWorkspaceRefreshKey = "";
  private boardRequestSeq = 0;
  private userNavigated = false;
  private closedTabIds = new Set<string>();
  private dockReturnFocus: HTMLElement | null = null;
  private pendingDeliveryFocusRunId: string | null = null;

  constructor(location: Location = window.location) {
    this.params = new URLSearchParams(location.search);
    const hashView = location.hash.replace(/^#/, "");
    const queryView = this.params.get("view");
    const queryTab = this.params.get("tab");
    const queryRun = this.params.get("run")?.trim() || null;
    this.sse = this.params.get("nosse") !== "1";
    const storedTabs = readStoredTabs();
    const initialTabs =
      queryRun && !storedTabs.includes(queryRun) ? [queryRun, ...storedTabs] : storedTabs;
    this.state = {
      activeSurface: queryRun ? "oracle" : "board",
      workspaces: [],
      selectedWorkspaceId: null,
      boardModeFilter: "all",
      selectedGodFilter: null,
      hoveredGod: null,
      showArchived: false,
      openTabs: initialTabs,
      quickDrawerRunId: null,
      newRunOpenSeq: 0,
      workspaceDialogOpenSeq: 0,
      allRuns: [],
      runs: [],
      pantheon: [],
      pendingRulingTotal: 0,
      runId: queryRun,
      projection: null,
      view: isView(hashView) ? hashView : isView(queryView) ? queryView : "topology",
      selectedTaskId: this.params.get("task") || null,
      workflowNode: null,
      deliveryEvidence: null,
      productInspectionRunId: null,
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

  private persistTabs(tabs: string[]): void {
    this.state.openTabs = tabs;
    writeStoredTabs(tabs);
  }

  private mergeAllRuns(incoming: RunSummary[], replaceArchived: boolean): void {
    if (replaceArchived) {
      this.state.allRuns = incoming;
      return;
    }
    const incomingIds = new Set(incoming.map((item) => item.run_id));
    const retainedArchived = this.state.allRuns.filter(
      (item) => item.is_archived && !incomingIds.has(item.run_id),
    );
    this.state.allRuns = [...incoming, ...retainedArchived];
  }

  async start(): Promise<void> {
    window.addEventListener("hashchange", () => {
      const view = window.location.hash.replace(/^#/, "");
      if (isView(view) && view !== this.state.view) this.setView(view);
    });
    this.setTheme(this.state.theme);
    const queryRun = this.params.get("run")?.trim() || null;

    let seeded = false;
    let [wsData, allData] = await Promise.all([
      listWorkspaces().catch(() => ({ ok: true as const, workspaces: [] })),
      listRuns({ includeArchived: true, limit: 200 }),
    ]);
    if (!allData.runs.length) {
      seeded = true;
      await seedDemo();
      [wsData, allData] = await Promise.all([
        listWorkspaces().catch(() => ({ ok: true as const, workspaces: [] })),
        listRuns({ includeArchived: true, limit: 200 }),
      ]);
    }

    const s = this.state;
    s.workspaces = wsData.workspaces || [];
    this.mergeAllRuns(allData.runs || [], true);
    s.pantheon = allData.pantheon || [];
    s.pendingRulingTotal = allData.pending_ruling_total ?? 0;
    s.runs = s.showArchived
      ? allData.runs || []
      : (allData.runs || []).filter((run) => !run.is_archived);

    const knownIds = new Set(s.allRuns.map((run) => run.run_id));
    const truncated = (allData.runs || []).length >= 200;
    const candidateTabs = s.openTabs.filter((id) => !this.closedTabIds.has(id));
    let validatedTabs: string[];
    if (!truncated) {
      validatedTabs = candidateTabs.filter((id) => knownIds.has(id) || id === queryRun);
    } else {
      const checks = await Promise.all(
        candidateTabs.map(async (id) =>
          knownIds.has(id) || id === queryRun ? true : await runExists(id),
        ),
      );
      validatedTabs = candidateTabs.filter((_, idx) => checks[idx]);
    }
    if (queryRun && !this.closedTabIds.has(queryRun) && !validatedTabs.includes(queryRun)) {
      validatedTabs.unshift(queryRun);
    }
    this.persistTabs(validatedTabs);

    if (this.userNavigated) {
      this.commit();
      return;
    }

    if (queryRun) {
      // Constructor already booted Oracle view on `queryRun` (streamGeneration 0); avoid
      // re-running `loadRun(queryRun)` after startup fetches finish, which would issue a
      // duplicate `/snapshot` fetch and race with `?nosse=1` stale-revision checks.
      if (s.activeSurface === "oracle" && s.runId === queryRun && !seeded) {
        this.syncUrl();
        this.commit();
      } else {
        s.activeSurface = "oracle";
        this.loadRun(queryRun);
      }
    } else {
      s.activeSurface = "board";
      s.runId = null;
      this.syncUrl();
      this.commit();
    }
  }

  async refreshWorkspaces(): Promise<WorkspaceSummary[]> {
    const data = await listWorkspaces();
    this.state.workspaces = data.workspaces || [];
    this.commit(false);
    return this.state.workspaces;
  }

  async refreshBoardRuns(): Promise<RunListResponse> {
    const seq = ++this.boardRequestSeq;
    const s = this.state;
    const data = await listRuns({
      workspaceId: s.selectedWorkspaceId,
      includeArchived: s.showArchived,
    });
    if (seq !== this.boardRequestSeq) return data;
    s.runs = data.runs || [];
    s.pantheon = data.pantheon || [];
    if (!s.selectedWorkspaceId) {
      s.pendingRulingTotal = data.pending_ruling_total ?? 0;
      this.mergeAllRuns(data.runs || [], s.showArchived);
    }
    this.commit();
    return data;
  }

  async refreshRuns(): Promise<RunListResponse> {
    const [allData, wsData] = await Promise.all([
      listRuns({ includeArchived: true, limit: 200 }),
      listWorkspaces().catch(() => null),
    ]);
    const s = this.state;
    if (wsData) s.workspaces = wsData.workspaces || [];
    this.mergeAllRuns(allData.runs || [], true);
    s.pendingRulingTotal = allData.pending_ruling_total ?? 0;
    if (!s.selectedWorkspaceId) {
      s.pantheon = allData.pantheon || [];
      s.runs = s.showArchived
        ? allData.runs || []
        : (allData.runs || []).filter((run) => !run.is_archived);
      this.commit();
      return allData;
    }
    return this.refreshBoardRuns();
  }

  applyBoardSnapshot = (snapshot: RunListResponse): void => {
    const s = this.state;
    const restore = s.activeSurface === "oracle" && s.projection ? captureReadingPosition() : null;
    this.mergeAllRuns(snapshot.runs || [], false);
    s.pendingRulingTotal = snapshot.pending_ruling_total ?? 0;

    if (!s.selectedWorkspaceId) {
      s.pantheon = snapshot.pantheon || [];
      if (!s.showArchived) {
        s.runs = snapshot.runs || [];
      } else {
        s.runs = s.allRuns;
        void this.refreshBoardRuns().catch(() => undefined);
      }
    } else {
      const source = s.showArchived ? s.allRuns : snapshot.runs || [];
      s.runs = source.filter((run) => run.workspace_id === s.selectedWorkspaceId);
      void this.refreshBoardRuns().catch(() => undefined);
    }

    const wsKey = `${snapshot.pending_ruling_total}:${(snapshot.runs || [])
      .map((run) => `${run.run_id}:${run.state_revision}:${run.ball_holder}:${run.workspace_id || ""}`)
      .join(",")}`;
    if (wsKey !== this.lastBoardWorkspaceRefreshKey) {
      this.lastBoardWorkspaceRefreshKey = wsKey;
      void this.refreshWorkspaces().catch(() => undefined);
    }

    this.commit();
    restore?.();
  };

  openBoard(options: { archived?: boolean } = {}): void {
    this.userNavigated = true;
    this.pendingDeliveryFocusRunId = null;
    const s = this.state;
    const nextArchived = options.archived ?? false;
    const archivedChanged = s.showArchived !== nextArchived;
    s.activeSurface = "board";
    s.showArchived = nextArchived;
    s.hoveredGod = null;
    s.dockOpen = false;
    this.syncUrl();
    if (archivedChanged || s.selectedWorkspaceId) {
      if (!s.selectedWorkspaceId) {
        s.runs = nextArchived
          ? s.allRuns
          : s.allRuns.filter((run) => !run.is_archived);
      } else {
        s.runs = s.allRuns.filter(
          (run) =>
            run.workspace_id === s.selectedWorkspaceId && (nextArchived || !run.is_archived),
        );
      }
      this.commit();
      void this.refreshBoardRuns().catch(() => undefined);
      return;
    }
    this.commit();
  }

  toggleArchive(): void {
    const s = this.state;
    if (s.activeSurface === "board" && s.showArchived) {
      this.openBoard({ archived: false });
    } else {
      this.openBoard({ archived: true });
    }
  }

  openOracleTab(runId: string): void {
    if (!runId) return;
    this.userNavigated = true;
    this.closedTabIds.delete(runId);
    const s = this.state;
    if (!s.openTabs.includes(runId)) {
      this.persistTabs([...s.openTabs, runId]);
    }
    s.activeSurface = "oracle";
    s.hoveredGod = null;
    s.quickDrawerRunId = null;
    if (s.runId !== runId || !s.projection || s.projection.run_id !== runId) {
      s.selectedTaskId = null;
      s.workflowNode = null;
      this.loadRun(runId);
    } else {
      s.dockOpen = s.userOpenedDock || s.projection.hitl_banner?.kind === "task_blocked";
      this.syncUrl();
      this.commit();
    }
  }

  openDeliveryEvidence(runId: string): void {
    if (!runId) return;
    this.state.view = "topology";
    this.pendingDeliveryFocusRunId = runId;
    this.openOracleTab(runId);
    const s = this.state;
    if (s.runId === runId && s.projection?.run_id === runId) {
      this.pendingDeliveryFocusRunId = null;
      if (s.projection.delivery_acceptance?.product_verification || s.projection.post_dag) {
        this.showProductEvidence();
      }
      this.focusDeliverySection();
    }
  }

  private focusDeliverySection(): void {
    const target =
      byId<HTMLElement>("product-verification-card") ||
      byId<HTMLElement>("delivery-acceptance") ||
      byId<HTMLElement>("topology-title");
    if (!target) return;
    target.scrollIntoView({ block: "start" });
    if (!this.state.dockOpen) {
      if (!target.hasAttribute("tabindex")) target.tabIndex = -1;
      target.focus();
    }
  }

  closeOracleTab(runId: string): void {
    this.userNavigated = true;
    this.closedTabIds.add(runId);
    const s = this.state;
    const idx = s.openTabs.indexOf(runId);
    if (idx === -1) return;
    const nextTabs = s.openTabs.filter((id) => id !== runId);
    this.persistTabs(nextTabs);

    if (s.activeSurface === "oracle" && s.runId === runId) {
      const nextRunId = nextTabs[Math.min(idx, nextTabs.length - 1)] || null;
      if (nextRunId) {
        s.selectedTaskId = null;
        s.workflowNode = null;
        this.loadRun(nextRunId);
      } else {
        s.runId = null;
        s.projection = null;
        s.dockOpen = false;
        s.userOpenedDock = false;
        s.activeSurface = "board";
        this.syncUrl();
        this.commit();
      }
      return;
    }
    if (s.runId === runId) {
      s.runId = null;
      s.projection = null;
    }
    this.commit();
  }

  selectWorkspace(workspaceId: string | null): void {
    const next = workspaceId && workspaceId.trim() ? workspaceId : null;
    const s = this.state;
    if (s.selectedWorkspaceId === next) return;
    s.selectedWorkspaceId = next;
    s.runs = s.allRuns.filter(
      (run) =>
        (next === null || run.workspace_id === next) && (s.showArchived || !run.is_archived),
    );
    this.commit();
    void this.refreshBoardRuns().catch(() => undefined);
  }

  setBoardModeFilter(mode: BoardModeFilter): void {
    if (this.state.boardModeFilter === mode) return;
    this.state.boardModeFilter = mode;
    this.commit();
  }

  setSelectedGodFilter(god: PantheonGod | null): void {
    const next = god && this.state.selectedGodFilter === god ? null : god;
    this.state.selectedGodFilter = next;
    if (next === null) {
      this.state.hoveredGod = null;
    }
    this.commit();
  }

  setGodFilter(god: PantheonGod | null): void {
    this.setSelectedGodFilter(god);
  }

  setHoveredGod(god: PantheonGod | null): void {
    if (this.state.hoveredGod === god) return;
    this.state.hoveredGod = god;
    this.commit();
  }

  openQuickDrawer(runId: string): void {
    this.state.quickDrawerRunId = runId;
    this.commit();
  }

  closeQuickDrawer(): void {
    if (this.state.quickDrawerRunId === null) return;
    this.state.quickDrawerRunId = null;
    this.commit();
  }

  async openWorkspaceManager(): Promise<void> {
    this.state.workspaceDialogOpenSeq += 1;
    this.commit();
    byId<HTMLDialogElement>("workspace-manager-dialog")?.showModal();
    try {
      await this.refreshWorkspaces();
    } catch {
      this.toast(tr("加载工作区列表失败"));
    }
  }

  async registerWorkspace(
    body: CreateWorkspaceBody,
  ): Promise<{ ok: true; workspace: WorkspaceSummary } | { ok: false; error: string }> {
    try {
      const response = await createWorkspace(body);
      await this.refreshWorkspaces();
      this.toast(tr("已注册工作区 {name}", { name: response.workspace.name }));
      return { ok: true, workspace: response.workspace };
    } catch (error) {
      return { ok: false, error: (error as Error).message || tr("注册工作区失败") };
    }
  }

  async cloneWorkspace(
    body: CloneWorkspaceBody,
  ): Promise<{ ok: true; workspace: WorkspaceSummary } | { ok: false; error: string }> {
    try {
      const response = await cloneWorkspace(body);
      await this.refreshWorkspaces();
      await this.refreshRuns().catch(() => undefined);
      this.toast(tr("已克隆并注册工作区 {name}", { name: response.workspace.name }));
      return { ok: true, workspace: response.workspace };
    } catch (error) {
      return { ok: false, error: (error as Error).message || tr("克隆工作区失败") };
    }
  }

  async unregisterWorkspace(workspaceId: string): Promise<boolean> {
    try {
      await deleteWorkspace(workspaceId);
      const resetFilter = this.state.selectedWorkspaceId === workspaceId;
      if (resetFilter) {
        this.state.selectedWorkspaceId = null;
      }
      await this.refreshWorkspaces();
      if (resetFilter) {
        await this.refreshBoardRuns();
      }
      this.toast(tr("已移除工作区注册（本地文件与历史记录均保留）"));
      return true;
    } catch (error) {
      this.toast(tr("移除工作区失败：{error}", { error: (error as Error).message }));
      return false;
    }
  }

  /** Switch the console to another run; the transport hook reconnects on `streamGeneration`. */
  loadRun(runId: string): void {
    const s = this.state;
    if (s.runId !== runId) {
      this.pendingAction = null;
      s.dialog = null;
      s.productInspectionRunId = null;
      if (this.pendingDeliveryFocusRunId && this.pendingDeliveryFocusRunId !== runId) {
        this.pendingDeliveryFocusRunId = null;
      }
      const hitlDialog = byId<HTMLDialogElement>("hitl-dialog");
      if (hitlDialog?.open) hitlDialog.close();
    }
    s.activeSurface = "oracle";
    if (!s.openTabs.includes(runId)) {
      this.persistTabs([...s.openTabs, runId]);
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
    this.openOracleTab(runId);
  }

  connectionText(): string {
    const s = this.state;
    return s.submitting ? tr("正在提交…") : (s.phase && PHASE_TEXT[s.phase]) || "";
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
    for (const list of [s.runs, s.allRuns]) {
      const runEntry = list.find((run) => run.run_id === projection.run_id);
      if (runEntry) {
        runEntry.status = projection.status;
        runEntry.workflow_name = projection.workflow_name || runEntry.workflow_name;
      }
    }
    if (switchedRun && !s.userOpenedDock) {
      s.dockOpen = projection.hitl_banner?.kind === "task_blocked";
    }
    this.syncApprovals(projection);
    const pendingDeliveryFocus = this.pendingDeliveryFocusRunId === projection.run_id;
    if (pendingDeliveryFocus) {
      this.pendingDeliveryFocusRunId = null;
      s.view = "topology";
      if (projection.delivery_acceptance?.product_verification || projection.post_dag) {
        s.deliveryEvidence = null;
        s.workflowNode = null;
        s.productInspectionRunId = projection.run_id;
        s.userOpenedDock = true;
        s.dockOpen = true;
      }
      this.syncUrl();
    }
    this.normalizeDock();
    this.commit();
    this.afterProjection(projection, previous);
    if (pendingDeliveryFocus) {
      this.focusDeliverySection();
      if (s.dockOpen) byId("dock-title")?.focus();
    } else {
      restore?.();
    }
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
    const next = buildApprovalSnapshot(projection, s.approvalSnapshot, s.approvalOrigins);
    if (!next) return;
    s.approvalOrigins = next.origins;
    s.approvalSnapshot = next.snapshot;
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
    if (
      s.productInspectionRunId &&
      (s.productInspectionRunId !== projection.run_id ||
        (!projection.delivery_acceptance?.product_verification && !projection.post_dag))
    ) {
      s.productInspectionRunId = null;
    }
    if (!s.deliveryEvidence && !s.workflowNode && !s.productInspectionRunId) {
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
    if (this.state.activeSurface === "oracle" && this.state.runId) {
      url.searchParams.set("run", this.state.runId);
      url.hash = this.state.view;
    } else {
      url.searchParams.delete("run");
      url.hash = "";
    }
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

  private rememberDockTrigger(): void {
    const active = document.activeElement;
    if (active instanceof HTMLElement && !active.closest("#dock") && active !== document.body) {
      this.dockReturnFocus = active;
    }
  }

  setDockOpen(open: boolean, byUser: boolean): void {
    if (open && byUser) this.rememberDockTrigger();
    this.state.userOpenedDock = byUser && open;
    this.state.dockOpen = open;
    this.commit();
  }

  closeDock(): void {
    const selection = this.state.deliveryEvidence;
    const returnEl = this.dockReturnFocus;
    this.dockReturnFocus = null;
    this.setDockOpen(false, false);
    const deliveryTrigger = selection
      ? document.querySelector<HTMLElement>(
          `#delivery-criteria [data-delivery-evidence="${selection.index}"]`,
        )
      : null;
    const isUsable = (el: HTMLElement | null): el is HTMLElement =>
      Boolean(el && el.isConnected && !el.closest("#dock") && !el.hidden && el.getClientRects().length);
    const target =
      (isUsable(returnEl) ? returnEl : null) ||
      (isUsable(deliveryTrigger) ? deliveryTrigger : null) ||
      byId<HTMLElement>("btn-dock-open");
    target?.focus();
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
    this.rememberDockTrigger();
    s.deliveryEvidence = null;
    s.productInspectionRunId = null;
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
    this.rememberDockTrigger();
    s.deliveryEvidence = null;
    s.productInspectionRunId = null;
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
    if (!s.projection || !s.runId || s.projection.run_id !== s.runId || !review) return;
    this.rememberDockTrigger();
    s.productInspectionRunId = null;
    s.workflowNode = null;
    s.deliveryEvidence = { index, specHash: review.spec_hash, runId: s.projection.run_id };
    s.userOpenedDock = true;
    s.dockOpen = true;
    this.normalizeDock();
    this.commit();
    byId("dock-title")?.focus();
  }

  showProductEvidence(): void {
    const s = this.state;
    if (
      !s.projection ||
      !s.runId ||
      s.projection.run_id !== s.runId ||
      (!s.projection.delivery_acceptance?.product_verification && !s.projection.post_dag)
    ) {
      return;
    }
    this.rememberDockTrigger();
    s.deliveryEvidence = null;
    s.workflowNode = null;
    s.productInspectionRunId = s.projection.run_id;
    s.userOpenedDock = true;
    s.dockOpen = true;
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
    if (!output || !s.projection || (s.runId && s.projection.run_id !== s.runId)) return;
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
    const latestProductSeq =
      s.projection.delivery_acceptance?.product_verification?.event_seq ?? null;
    if (
      output.event_type === "product.verified" &&
      latestProductSeq != null &&
      output.event_seq === latestProductSeq
    ) {
      this.showProductEvidence();
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
      this.toast(tr("读取执行产物失败，可重新打开"));
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
          title: tr("GoalSpec v{version} 仍有未决阻断项", { version: spec.version }),
          description: tr("请先处理阻断项。若要批准此版本，请填写接受风险的理由。"),
          label: tr("接受风险理由（必填）"),
          defaultValue: tr("首期范围已与业务确认，接受未决歧义风险先行冻结规格。"),
          requireNonEmpty: true,
          onConfirm: (reason) =>
            this.sendCommand(
              { ...command, action: "approve_spec", accept_risk: true, reason },
              tr("已带风险批准 GoalSpec，进入架构设计"),
              runId,
            ),
        });
        return;
      }
      await this.sendCommand(command, tr("已批准 GoalSpec，进入架构设计"), runId);
    } else if (command.action === "approve_design") {
      await this.sendCommand(command, tr("已批准 DesignSpec，启动任务分解"), runId);
    } else {
      const message =
        this.state.projection?.workflow?.stop_after === "plan"
          ? tr("已批准任务计划，蓝图编排已完成")
          : tr("已批准任务计划，开始执行");
      await this.sendCommand(command, message, runId);
    }
  }

  /** Post a command bound to the revision on screen; a 409 applies the fresh projection. */
  async sendCommand(body: CommandBody, successMessage: string, runId = this.state.runId): Promise<void> {
    const s = this.state;
    if (!runId || runId !== s.runId || s.projection?.run_id !== runId) return;
    const result = await executeRunCommand(runId, s.projection.state_revision, body);
    if (this.state.runId !== runId) return;
    if (result.kind === "network_error" || result.kind === "rejected") {
      this.toast(result.message);
      return;
    }
    if (result.kind === "conflict") {
      if (result.projection) {
        this.applyProjection(result.projection);
        this.refreshRuns().catch(() => undefined);
      }
      this.toast(result.message);
      return;
    }
    this.applyProjection(result.projection);
    this.refreshRuns().catch(() => undefined);
    this.toast(successMessage);
  }

  togglePause(): Promise<void> {
    const action = this.state.projection?.status === "paused" ? "resume" : "pause";
    return this.sendCommand(
      { action },
      action === "pause" ? tr("已暂停新任务派发") : tr("已恢复任务编排调度"),
    );
  }

  promoteToExecute(): Promise<void> {
    return this.sendCommand(
      { action: "promote_to_execute", workflow_name: "full_sdlc_tdd" },
      tr("已升级为 full_sdlc_tdd，启动沙盒代码执行"),
    );
  }

  retryWorkflow(): Promise<void> {
    return this.sendCommand({ action: "retry_workflow" }, tr("已从断点重新启动工作流"));
  }

  requestRevision(feedback: string): Promise<void> {
    return this.sendCommand(
      { action: "request_revision", feedback: feedback.trim() },
      tr("已提交修订请求，工作流从需求对齐阶段重新启动"),
    );
  }

  openReopenRun(): void {
    const projection = this.state.projection;
    if (!projection) return;
    const targetRunId = projection.run_id;
    this.openDialog({
      title: tr("追加工单"),
      description: tr(
        "在已完成的运行上追加后续需求或修复反馈；工作流会从需求对齐阶段重新开始，并保留原有约束与工作区范围。",
      ),
      label: tr("追加需求或修复反馈（必填）"),
      defaultValue: "",
      placeholder: tr("例如：补充导出 Excel 的功能，并保持现有 API 兼容。"),
      requireNonEmpty: true,
      onConfirm: (feedback) =>
        this.sendCommand(
          { action: "reopen_run", feedback: feedback.trim() },
          tr("已追加工单，工作流从需求对齐阶段重新启动"),
          targetRunId,
        ),
    });
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
      tr("已将任务 {taskId} 重置为就绪 (READY)", { taskId }),
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
      title: tr("向任务 {taskId} 注入人工提示并重试", { taskId }),
      description: tr(
        "提示原文保留在审计事件中，随下次隔离沙盒执行一并注入 Worker 上下文，不直接篡改需求契约。",
      ),
      label: tr("注入提示内容"),
      defaultValue: "",
      placeholder: tr("输入要注入的人工提示，例如需修复的边界或重试思路。"),
      requireNonEmpty: true,
      onConfirm: (hint) =>
        this.sendCommand(
          {
            action: "retry_task",
            task_id: taskId,
            hint,
            expected_state_revision: expectedRevision,
          },
          tr("已向 {taskId} 注入提示并重新排入就绪队列", { taskId }),
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
      title: tr("将任务 {taskId} 改派给人工工程师", { taskId }),
      description: tr(
        "改派后人工提交的产物仍须经过相同的程序门禁与 Argus 独立复核才能成为已验收状态。",
      ),
      label: tr("改派备注"),
      defaultValue: "",
      placeholder: tr("输入改派备注，例如接手人与需关注的验收要点。"),
      requireNonEmpty: true,
      onConfirm: (note) =>
        this.sendCommand(
          {
            action: "reassign_human",
            task_id: taskId,
            note,
            expected_state_revision: expectedRevision,
          },
          tr("已记录 {taskId} 改派人工并回到就绪态", { taskId }),
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
      title: tr("接受风险完成任务 {taskId}", { taskId }),
      description: tr(
        "必须填写明确理由。该决定与 Agent 建议分开存储，追踪矩阵标记为“未验证 · 人工接受”，绝不显示为“已验证”。",
      ),
      label: tr("接受风险理由（必填）"),
      defaultValue: "",
      placeholder: tr("输入接受风险的明确理由，例如影响范围与缓解措施。"),
      requireNonEmpty: true,
      onConfirm: (reason) =>
        this.sendCommand(
          {
            action: "accept_risk",
            task_id: taskId,
            reason,
            expected_state_revision: expectedRevision,
          },
          tr("已记录人工接受风险，{taskId} 标记为未验证 · 人工接受", { taskId }),
          targetRunId,
        ),
    });
  }

  async submitClarify(answers: Record<string, string>, clarificationNote = ""): Promise<void> {
    const projection = this.state.projection;
    if (!projection) return;
    const note = clarificationNote.trim();
    if (!Object.keys(answers).length && !note) {
      this.toast(tr("请先回答至少一项澄清问题或填写补充说明"));
      return;
    }
    const remaining = pendingAmbiguities(projection).filter(
      (amb) => amb.blocking && !answers[amb.id],
    );
    this.state.clarifyGeneration += 1;
    await this.sendCommand(
      { action: "clarify", answers, clarification_note: note },
      remaining.length
        ? tr("已保存澄清进度，仍可继续回答剩余阻断项")
        : tr("已提交澄清并生成新的 GoalSpec 版本"),
    );
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
      this.toast(tr("必须填写理由或提示后才能提交"));
      return;
    }
    await pending.onConfirm(text);
  }

  async openNewRun(): Promise<void> {
    this.state.workflowNode = null;
    this.state.newRunOpenSeq += 1;
    this.commit();
    const dialog = byId<HTMLDialogElement>("new-run-dialog");
    if (dialog && !dialog.open) dialog.showModal();
    try {
      await Promise.all([
        this.loadNewRunOptions(),
        this.refreshWorkspaces().catch(() => undefined),
      ]);
    } catch {
      this.toast(tr("加载工作流或提供方失败"));
    }
  }

  private async loadNewRunOptions(): Promise<void> {
    if (this.state.options) return;
    this.state.options = await fetchWorkbenchOptions();
    this.commit(false);
  }

  async submitNewRun(form: NewRunForm): Promise<NewRunSubmitResult> {
    const s = this.state;
    const rawInput = form.input.trim();
    if (!rawInput) {
      return { ok: false, error: tr("请先填写需求原文或导入文件") };
    }
    const demoId = form.demoId.trim();
    const selectedDemo = demoId ? s.options?.demos.find((item) => item.id === demoId) : undefined;
    const mode = demoId ? "brownfield" : form.workspaceMode;

    if (!demoId && mode === "greenfield" && !form.greenfieldPath.trim()) {
      return { ok: false, error: tr("Greenfield 模式必须填写目标目录路径") };
    }
    if (
      !demoId &&
      mode === "brownfield" &&
      form.workspaceId === "__custom__" &&
      !form.customWorkspaceRoot.trim()
    ) {
      return { ok: false, error: tr("请填写现有本地仓库目录路径") };
    }

    const provider = form.provider || "auto";
    const model = form.model.trim();
    const providerConfig: NonNullable<CreateRunBody["provider_config"]> =
      provider === "auto" ? {} : { provider };
    if (model) providerConfig.model = model;

    const payload: CreateRunBody = {
      raw_input: rawInput,
      title: form.title.trim() || rawInput.slice(0, 60),
      workspace_mode: mode,
      task_type: form.taskType,
      provider_config: providerConfig,
    };

    const workflowOverride = form.workflow.trim();
    if (workflowOverride) {
      payload.workflow_name = workflowOverride;
    }

    const overrides = [
      ["worker_driver", form.workerDriver],
      ["reviewer_driver", form.reviewerDriver],
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
      if (!workflowOverride && selectedDemo?.default_workflow) {
        payload.workflow_name = selectedDemo.default_workflow;
      }
    } else if (mode === "brownfield") {
      if (form.workspaceId === "__custom__") {
        payload.workspace_root = form.customWorkspaceRoot.trim();
      } else if (form.workspaceId.trim()) {
        payload.workspace_id = form.workspaceId.trim();
      } else if (s.selectedWorkspaceId) {
        payload.workspace_id = s.selectedWorkspaceId;
      }
      const baseRef = form.baseRef.trim();
      if (baseRef) payload.base_ref = baseRef;
      const scopePaths = form.scopePaths
        .split(/[\n,]+/)
        .map((item) => item.trim())
        .filter(Boolean);
      if (scopePaths.length) payload.scope_paths = scopePaths;
    } else if (mode === "greenfield") {
      payload.workspace_root = form.greenfieldPath.trim();
      if (form.initGit) {
        payload.init_git = true;
      }
    } else if (mode === "spec_only") {
      if (form.contextWorkspaceId.trim()) {
        payload.workspace_id = form.contextWorkspaceId.trim();
      }
    }

    if (s.submitting) return { ok: false, error: tr("正在提交中，请稍候") };
    s.submitting = true;
    this.renderConnection();
    let runId: string;
    try {
      runId = (await createRun(payload)).run_id;
    } catch (error) {
      return { ok: false, error: (error as Error).message || tr("未知错误") };
    } finally {
      s.submitting = false;
      this.renderConnection();
    }

    s.selectedTaskId = null;
    s.workflowNode = null;
    await this.refreshRuns().catch(() => undefined);
    if (s.activeSurface === "board") {
      const created = s.allRuns.find((run) => run.run_id === runId);
      if (s.selectedWorkspaceId && created && created.workspace_id !== s.selectedWorkspaceId) {
        this.selectWorkspace(created.workspace_id || null);
      }
    } else {
      this.loadRun(runId);
    }
    this.toast(tr("已接收编排 {runId}", { runId }));
    return { ok: true, runId };
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
