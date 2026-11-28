import type { RunEvent } from "../api/types";
import { indexIncidents, positiveAttempt } from "./taskIncidents";
import { indexMilestones } from "./replayMilestones";

interface ObservedTask {
  id: string;
  actor: string;
  status: string;
  attempt?: number;
  seq: number;
}

const STATES = new Set(["waiting", "ready", "running", "verifying", "accepted", "failed", "blocked", "cancelled", "stale"]);
const CHECKPOINT_INTERVAL = 100;

// Swimlane entries describe one actor's observation. A review lane can report an
// accepted review while the worker lane still reports a blocked task.
function observation(event: RunEvent, prior?: ObservedTask): ObservedTask | undefined {
  const p = event.payload ?? {};
  if (typeof p.task_id !== "string" || !p.task_id) return;
  let status: unknown;
  let attempt = p.attempt;
  if (event.event_type === "agent.swimlane") {
    status = p.status;
    attempt = p.attempts;
  } else if (event.event_type === "task.leased") {
    status = "running";
  } else if (event.event_type === "task.retried_by_human" || event.event_type === "task.reassigned_human") {
    status = "ready";
  } else if (event.event_type === "task.accepted_with_risk") {
    status = "accepted_with_risk";
  } else if (event.event_type.startsWith("task.")) {
    status = p.status;
  }
  if (typeof status !== "string" || (!STATES.has(status) && status !== "accepted_with_risk")) return;
  return {
    id: p.task_id, actor: event.actor, status, seq: event.event_seq,
    attempt: attempt == null ? prior?.attempt : positiveAttempt(attempt),
  };
}

function fold(tasks: Map<string, ObservedTask>, event: RunEvent) {
  const id = event.payload?.task_id;
  if (typeof id !== "string") return;
  const key = JSON.stringify([id, event.event_type === "agent.swimlane" ? event.actor : null]);
  const next = observation(event, tasks.get(key));
  if (next) tasks.set(key, next);
  return next;
}

export function indexReplay(runId: string, horizon: number, history: RunEvent[]) {
  const events = history.filter(e => e.run_id === runId && e.event_seq <= horizon);
  // GET /events is ordered by a gapless run-local sequence, starting at 1.
  if (events.length !== horizon || events.some((e, i) => e.event_seq !== i + 1)) {
    throw new Error("Incomplete event history");
  }
  const tasks = new Map<string, ObservedTask>();
  const checkpoints = [new Map(tasks)];
  events.forEach((event, index) => {
    fold(tasks, event);
    if ((index + 1) % CHECKPOINT_INTERVAL === 0) checkpoints.push(new Map(tasks));
  });
  return {
    events, incidentsAt: indexIncidents(events), milestonesAt: indexMilestones(events),
    tasksAt(cursor: number) {
      const block = Math.floor(cursor / CHECKPOINT_INTERVAL);
      const current = new Map(checkpoints[block]);
      for (let i = block * CHECKPOINT_INTERVAL; i < cursor; i++) fold(current, events[i]);
      return [...current.values()].sort((a, b) => b.seq - a.seq);
    },
  };
}
