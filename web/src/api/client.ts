import type {
  AttemptArtifactsResponse,
  CommandBody,
  CommandResponse,
  ConflictResponse,
  CreateRunBody,
  CreateRunResponse,
  DemosResponse,
  ErrorResponse,
  ProvidersResponse,
  RecordedAttemptArtifacts,
  RunListResponse,
  RunProjection,
  RunProjectionResponse,
  WorkflowsResponse,
} from "./types";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const runPath = (runId: string): string => `/api/runs/${encodeURIComponent(runId)}`;

export const streamUrl = (runId: string, afterSeq: number): string =>
  `${runPath(runId)}/stream?after_seq=${afterSeq}`;

export const deliveryLogUrl = (runId: string, eventSeq: number, position: number): string =>
  `${runPath(runId)}/delivery-log/${eventSeq}/${position}`;

async function json<T>(response: Response): Promise<T> {
  return (await response.json()) as T;
}

export async function listRuns(): Promise<RunListResponse> {
  return json<RunListResponse>(await fetch("/api/runs"));
}

export async function seedDemo(): Promise<void> {
  await fetch("/api/seed-demo", { method: "POST", body: "{}" });
}

export async function fetchSnapshot(runId: string): Promise<RunProjection> {
  const response = await fetch(runPath(runId), { cache: "no-store" });
  const data = await json<RunProjectionResponse | ErrorResponse>(response);
  if (!response.ok || !data.ok || data.projection.run_id !== runId) {
    throw new ApiError((!data.ok && data.error) || "加载运行快照失败", response.status);
  }
  return data.projection;
}

export async function createRun(body: CreateRunBody): Promise<CreateRunResponse> {
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await json<CreateRunResponse | ErrorResponse>(response);
  if (!data.ok || !data.run_id) throw new ApiError(data.error || "未知错误", response.status);
  return data;
}

export type CommandOutcome =
  | { kind: "applied"; projection: RunProjection }
  | { kind: "conflict"; projection: RunProjection | null }
  | { kind: "rejected"; error: string };

export async function postCommand(runId: string, body: CommandBody): Promise<CommandOutcome> {
  const response = await fetch(`${runPath(runId)}/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await json<CommandResponse | ConflictResponse | ErrorResponse>(response);
  if (data.ok) return { kind: "applied", projection: data.projection };
  if (response.status === 409 || "conflict" in data) {
    let fresh = "projection" in data ? data.projection ?? null : null;
    if (!fresh) {
      try {
        const latest = await json<RunProjectionResponse | ErrorResponse>(await fetch(runPath(runId)));
        if (latest.ok) fresh = latest.projection;
      } catch {
        fresh = null;
      }
    }
    return { kind: "conflict", projection: fresh };
  }
  return { kind: "rejected", error: data.error || "未知错误" };
}

export interface WorkbenchOptions {
  workflows: WorkflowsResponse["workflows"];
  providers: ProvidersResponse["providers"];
  demos: DemosResponse["demos"];
}

export async function fetchWorkbenchOptions(): Promise<WorkbenchOptions> {
  const [workflows, providers, demos] = await Promise.all([
    fetch("/api/workflows").then((response) => json<WorkflowsResponse>(response)),
    fetch("/api/providers").then((response) => json<ProvidersResponse>(response)),
    fetch("/api/demos").then((response) => json<DemosResponse>(response)),
  ]);
  return {
    workflows: workflows.workflows || [],
    providers: providers.providers || [],
    demos: demos.demos || [],
  };
}

export async function fetchAttemptArtifacts(
  runId: string,
  taskId: string,
  attempt: number,
): Promise<RecordedAttemptArtifacts> {
  const url = `${runPath(runId)}/task-artifacts/${encodeURIComponent(taskId)}/${attempt}`;
  const response = await fetch(url);
  const data = await json<AttemptArtifactsResponse | ErrorResponse>(response);
  if (!response.ok || !data.ok) {
    throw new ApiError((!data.ok && data.error) || "读取执行产物失败", response.status);
  }
  return data.artifacts;
}
