/**
 * Snapshot + SSE transport for one run.
 *
 * The SSE `state` frame is only a change signal: every frame that advances the revision or event
 * sequence triggers an authoritative snapshot fetch. Transport loss keeps the last confirmed
 * projection on screen, reconnects after a short delay, and resumes from the newest event seen.
 */
import { useEffect, useRef } from "react";

import { ApiError, fetchSnapshot, streamUrl } from "./client";
import type { ActivityPreviews, RunProjection, StreamActivityFrame, StreamStateFrame } from "./types";

export type StreamPhase = "loading" | "online" | "reconnecting" | "missing";

export interface RunStreamHandlers {
  current: () => RunProjection | null;
  /** Offer a snapshot of this run; the console decides whether it is newer than what it shows. */
  apply: (projection: RunProjection) => void;
  phase: (phase: StreamPhase) => void;
  activity: (previews: ActivityPreviews | undefined) => void;
  missing: (message: string) => void;
}

export interface RunStreamOptions {
  /** `false` (or `?nosse=1`) loads one snapshot without opening an EventSource. */
  sse: boolean;
  reconnectDelayMs?: number;
}

export function openRunStream(
  runId: string,
  handlers: RunStreamHandlers,
  options: RunStreamOptions,
): () => void {
  let disposed = false;
  let source: EventSource | null = null;
  let timer: number | null = null;
  const delay = options.reconnectDelayMs ?? 1200;

  const latestSeq = (fallback: number): number => {
    const current = handlers.current();
    return current && current.run_id === runId ? current.latest_event_seq : fallback;
  };

  const offer = (projection: RunProjection): void => {
    if (!disposed && projection.run_id === runId) handlers.apply(projection);
  };

  const scheduleReconnect = (): void => {
    source?.close();
    source = null;
    handlers.activity(undefined);
    handlers.phase("reconnecting");
    if (timer !== null) return;
    timer = window.setTimeout(() => {
      timer = null;
      if (!disposed) void syncAndConnect();
    }, delay);
  };

  const connect = (afterSeq: number): void => {
    if (typeof EventSource === "undefined" || !options.sse) {
      handlers.phase("online");
      return;
    }
    const stream = new EventSource(streamUrl(runId, afterSeq));
    source = stream;
    const isCurrent = (): boolean => !disposed && source === stream;
    let requestedRevision = -1;
    let requestedSeq = afterSeq;

    stream.addEventListener("state", async (event: MessageEvent<string>) => {
      if (!isCurrent()) return;
      let request: [number, number] | undefined;
      try {
        const frame = JSON.parse(event.data) as StreamStateFrame;
        if (frame.run_id !== runId) return;
        const revision = frame.state_revision;
        const seq = frame.events.at(-1)?.event_seq ?? afterSeq;
        const current = handlers.current();
        const shownRevision = current?.run_id === runId ? current.state_revision : -1;
        const shownSeq = current?.run_id === runId ? current.latest_event_seq : -1;
        if (revision <= shownRevision && seq <= shownSeq) {
          if (requestedRevision <= shownRevision && requestedSeq <= shownSeq) {
            handlers.phase("online");
          }
          return;
        }
        if (revision <= requestedRevision && seq <= requestedSeq) return;
        requestedRevision = Math.max(requestedRevision, revision);
        requestedSeq = Math.max(requestedSeq, seq);
        request = [requestedRevision, requestedSeq];
        const projection = await fetchSnapshot(runId);
        if (!isCurrent()) return;
        offer(projection);
        const shown = handlers.current();
        if (
          shown &&
          shown.state_revision >= requestedRevision &&
          shown.latest_event_seq >= requestedSeq
        ) {
          handlers.phase("online");
          return;
        }
        if (request[0] !== requestedRevision || request[1] !== requestedSeq) return;
      } catch {
        if (
          !isCurrent() ||
          (request && (request[0] !== requestedRevision || request[1] !== requestedSeq))
        ) {
          return;
        }
      }
      if (isCurrent()) scheduleReconnect();
    });

    stream.addEventListener("activity", (event: MessageEvent<string>) => {
      if (!isCurrent()) return;
      const frame = JSON.parse(event.data) as StreamActivityFrame;
      if (frame.run_id !== runId) return;
      handlers.activity(frame.previews);
    });

    stream.addEventListener("error", () => {
      if (isCurrent()) scheduleReconnect();
    });
  };

  async function syncAndConnect(): Promise<void> {
    try {
      const projection = await fetchSnapshot(runId);
      if (disposed) return;
      offer(projection);
      connect(latestSeq(projection.latest_event_seq));
    } catch (error) {
      if (disposed) return;
      if (error instanceof ApiError && error.status === 404) {
        handlers.phase("missing");
        handlers.missing(error.message);
      } else {
        scheduleReconnect();
      }
    }
  }

  const onOffline = (): void => {
    if (source) scheduleReconnect();
  };

  window.addEventListener("offline", onOffline);
  // Start outside the caller's (possibly React effect) stack so handlers may commit synchronously.
  queueMicrotask(() => {
    if (disposed) return;
    handlers.phase("loading");
    void syncAndConnect();
  });

  return () => {
    disposed = true;
    window.removeEventListener("offline", onOffline);
    source?.close();
    source = null;
    if (timer !== null) window.clearTimeout(timer);
  };
}

/**
 * Keep one transport open for the selected run. `generation` re-opens the stream for the same
 * run id (for example after creating a run); handlers are read through a ref so callers may pass
 * fresh closures on every render without reconnecting.
 */
export function useRunStream(
  runId: string | null,
  generation: number,
  handlers: RunStreamHandlers,
  options: RunStreamOptions,
): void {
  const latest = useRef(handlers);
  latest.current = handlers;
  const { sse, reconnectDelayMs } = options;

  useEffect(() => {
    if (!runId) return undefined;
    return openRunStream(
      runId,
      {
        current: () => latest.current.current(),
        apply: (projection) => latest.current.apply(projection),
        phase: (phase) => latest.current.phase(phase),
        activity: (previews) => latest.current.activity(previews),
        missing: (message) => latest.current.missing(message),
      },
      { sse, reconnectDelayMs },
    );
  }, [runId, generation, sse, reconnectDelayMs]);
}
