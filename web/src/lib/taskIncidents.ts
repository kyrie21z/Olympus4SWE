import type { RunEvent } from "../api/types";

type Note = { seq: number; actor: string; label: string };
export type AttemptRef = { number: number; seq: number };
export interface TaskIncident {
  id: string;
  failure: Note;
  reason?: string;
  attempt?: number;
  previousFailure?: AttemptRef;
  nextAttempt?: AttemptRef;
  intervention?: Note;
  restart?: Note;
  result?: Note;
}

export function positiveAttempt(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

export function taskFailureLabel(event: RunEvent) {
  const p = event.payload ?? {};
  if (event.event_type === "task.diff_gate_rejected") return "文件变更未通过范围检查";
  if (event.event_type === "task.escape_hatch_rejected") return "代码变更未通过门禁检查";
  if (event.event_type === "review.verdict_recorded" && p.passed === false) return "独立审查未通过";
  if (event.event_type === "task.failed" || event.event_type === "task.blocked" ||
      (event.event_type === "agent.swimlane" && (p.status === "blocked" || p.status === "failed"))) {
    return "任务执行受阻";
  }
}

function summaryText(value: unknown) {
  if (typeof value !== "string" || !value.trim()) return;
  // Only show short recorded summaries. Omit credential-shaped or multiline content.
  if (/[\r\n]|bearer\s|api[_ -]?key|token\s*[:=]|password|secret|sk-[\w-]{8,}|gh[pousr]_/i.test(value)) return;
  return value.trim().slice(0, 180);
}

/** Store a version only when a task's incident changes; scrubbing looks up that prefix. */
export function indexIncidents(events: RunEvent[]) {
  const byTask = new Map<string, { seq: number; card: TaskIncident }[]>();
  function record(seq: number, card: TaskIncident) {
    const versions = byTask.get(card.id);
    if (versions) versions.push({ seq, card });
    else byTask.set(card.id, [{ seq, card }]);
  }
  const attempts = new Map<string, number>();
  for (const event of events) {
    const p = event.payload ?? {};
    const id = p.task_id;
    if (typeof id !== "string" || !id) continue;
    const kind = event.event_type;
    const seq = event.event_seq;
    const actor = event.actor;
    const raw = kind === "agent.swimlane" ? p.attempts : p.attempt;
    const count = positiveAttempt(raw);
    if (kind === "task.leased") {
      if (count === undefined) attempts.delete(id);
      else attempts.set(id, count);
    }
    const attempt = raw == null ? attempts.get(id) : count;
    const incident = byTask.get(id)?.at(-1)?.card;
    const failure = taskFailureLabel(event);
    if (failure) {
      if (attempt && incident?.attempt && attempt < incident.attempt) continue;
      // A terminal blocked event can follow the specific rejection in the same attempt.
      if (kind === "task.blocked" && incident && attempt !== undefined && attempt === incident.attempt &&
          !incident.result && !incident.restart && !incident.intervention) continue;
      const previousFailure = attempt && incident?.attempt && incident.attempt < attempt
        ? { number: incident.attempt, seq: incident.failure.seq } : incident?.previousFailure;
      record(seq, { id, attempt, previousFailure, failure: { seq, actor, label: failure },
        reason: kind === "agent.swimlane" ? summaryText(p.summary) : undefined });
      continue;
    }
    if (!incident || incident.result || (incident.attempt !== undefined && attempt !== undefined && attempt < incident.attempt)) continue;
    const update: Partial<TaskIncident> = {};
    if (!incident.nextAttempt && incident.attempt && count && count > incident.attempt &&
        (kind === "task.leased" || kind === "review.verdict_recorded" || kind === "task.accepted")) {
      update.nextAttempt = { number: count, seq };
    }
    const note = (label: string): Note => ({ label, seq, actor });
    if (kind === "task.retried_by_human") update.intervention = note("人工已请求重试");
    if (kind === "task.requeued") update.intervention = note("已重新排队");
    if (kind === "task.reassigned_human") update.intervention = note("已改派人工");
    if (kind === "task.leased") update.restart = note("已获得新的执行租约");
    if (kind === "task.accepted") update.result = note("任务已验收");
    if (kind === "task.accepted_with_risk") update.result = note("人工接受风险放行");
    if (Object.keys(update).length) record(seq, { ...incident, ...update });
  }
  return (cursor: number) => {
    const cards: TaskIncident[] = [];
    for (const versions of byTask.values()) {
      let lo = 0;
      let hi = versions.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (versions[mid].seq <= cursor) lo = mid + 1;
        else hi = mid;
      }
      if (lo) cards.push(versions[lo - 1].card);
    }
    return cards.sort((a, b) =>
      Number(Boolean(a.result)) - Number(Boolean(b.result)) || b.failure.seq - a.failure.seq);
  };
}
