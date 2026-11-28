import { useEffect, useMemo, useRef, useState } from "react";

import { fetchRunEvents } from "../api/client";
import { AttemptEvidence } from "./AttemptEvidence";
import { useT } from "../i18n";
import { statusMeta } from "../lib/labels";
import { publicId } from "../lib/evidenceId";
import { indexReplay } from "../lib/taskReplay";
import type { Workbench } from "../state/workbench";
import "../styles/replay.css";

interface Props {
  runId: string;
  latestSeq: number;
  workbench: Workbench;
}

export function TaskReplay({ runId, latestSeq, workbench }: Props) {
  const t = useT();
  const [horizon, setHorizon] = useState<number | null>(null);
  return <>
    <button className="btn" type="button" onClick={() => setHorizon(latestSeq)}>{t("故障复盘")}</button>
    {horizon !== null && <ReplayDialog runId={runId} horizon={horizon} liveSeq={latestSeq} onExit={() => {
      setHorizon(null);
      if (workbench.state.runId === runId) workbench.loadRun(runId);
    }} />}
  </>;
}

function ReplayDialog({ runId, horizon, liveSeq, onExit }: {
  runId: string; horizon: number; liveSeq: number; onExit: () => void;
}) {
  const t = useT();
  const dialog = useRef<HTMLDialogElement>(null);
  const sourceToggle = useRef<HTMLButtonElement>(null);
  const [index, setIndex] = useState<ReturnType<typeof indexReplay> | null>(null);
  const [cursor, setCursor] = useState(0);
  const [raw, setRaw] = useState(false);
  const [failed, setFailed] = useState(false);
  const [request, setRequest] = useState(0);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  useEffect(() => {
    const abort = new AbortController();
    setFailed(false);
    fetchRunEvents(runId, abort.signal).then(events => {
      if (abort.signal.aborted) return;
      const loaded = indexReplay(runId, horizon, events);
      setIndex(loaded);
      setCursor(horizon);
    }).catch(() => { if (!abort.signal.aborted) setFailed(true); });
    return () => abort.abort();
  }, [runId, horizon, request]);
  const tasks = useMemo(() => index?.tasksAt(cursor) ?? [], [index, cursor]);
  const through = raw ? cursor : horizon;
  const incidents = useMemo(() => index?.incidentsAt(through) ?? [], [index, through]);
  function openSource(seq: number) {
    setCursor(seq);
    setRaw(true);
    sourceToggle.current?.focus();
  }
  const event = index?.events[cursor - 1];
  const milestones = raw ? index?.milestonesAt(cursor) : undefined;
  const currentStop = milestones?.current;
  const step = (delta: number) => setCursor(value => Math.max(0, Math.min(value + delta, horizon)));
  return (
    <dialog id="run-replay" ref={dialog} aria-labelledby="replay-title" onClose={onExit}
      onKeyDown={e => {
        if (!index || !raw || e.altKey || e.ctrlKey || e.metaKey) return;
        if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
          const forward = e.key === "ArrowRight";
          if (e.shiftKey) {
            e.preventDefault();
            const destination = forward ? milestones?.next : milestones?.previous;
            if (destination !== undefined) setCursor(destination);
            return;
          }
          if (e.target instanceof HTMLElement && e.target.closest("input")) return;
          e.preventDefault();
          step(forward ? 1 : -1);
        }
      }}>
      <header className="replay-head">
        <div><span className="badge" data-variant="live">{t("只读回放")}</span>
          <h2 id="replay-title" className="dialog-title">{t("故障复盘")}</h2><code>{runId}</code></div>
        <button type="button" className="btn" onClick={() => dialog.current?.close()}>{t("回到实时")}</button>
      </header>
      {failed ? <div role="alert">{t("未能读取完整历史。")}
        <button type="button" className="btn" onClick={() => setRequest(n => n + 1)}>{t("重试")}</button></div>
        : !index ? <p role="status">{t("正在读取事件…")}</p>
        : <>
          <div className="replay-row replay-summary">
            <strong>{t("涉及 {n} 个故障任务", { n: incidents.length })}</strong>
            <small>{t("截至事件 {n}", { n: through })}</small>
            {liveSeq > horizon && <span className="badge" data-variant="live">{t("实时新增 {n} 条", { n: liveSeq - horizon })}</span>}
          </div>
          {!incidents.length && <p>{t("这段历史中没有记录到任务故障。")}</p>}
          {incidents.map(item => <article key={item.id} className="incident-card" aria-label={item.id}>
            <header className="replay-row"><strong>{item.id}</strong>
              <span className="badge" data-variant={item.result?.label === "任务已验收" ? "ok" : "warn"}>
                {t(item.result?.label ?? "尚无验收结果")}
              </span>
            </header>
            <h3>{item.reason ?? t(item.failure.label)}</h3>
            <p>{item.failure.actor}{item.attempt && <> · {t("第 {n} 次尝试", { n: item.attempt })}</>}</p>
            <dl>
              <dt>{t("故障证据")}</dt><dd><button className="btn" onClick={() => openSource(item.failure.seq)}>
                {t(item.failure.label)} · #{item.failure.seq}</button></dd>
              <dt>{t("后续处理")}</dt><dd>{!item.intervention && !item.restart && t("暂无重试或改派记录")}
                {[item.intervention, item.restart].map(note => note && <button key={note.seq} className="btn" onClick={() => {
                  openSource(note.seq);
                }}>{t(note.label)} · {note.actor} · #{note.seq}</button>)}</dd>
              <dt>{t("验证结果")}</dt><dd>{item.result ? <button className="btn" onClick={() => {
                openSource(item.result!.seq);
              }}>{t(item.result.label)} · #{item.result.seq}</button> : t("尚未记录该任务后续通过验收")}</dd>
            </dl>
            {through === horizon && <AttemptEvidence runId={runId} incident={item} onEvent={openSource} />}
          </article>)}
          <button ref={sourceToggle} type="button" className="btn" aria-expanded={raw} aria-controls="replay-source"
            onClick={() => setRaw(value => !value)}>{t(raw ? "收起原始事件" : "查看原始事件")}</button>
          <div id="replay-source" hidden={!raw}>{raw && <>
          <section className="replay-controls" aria-label={t("回放位置")}>
            <div className="replay-row">
              <strong>{t("事件序号")} <output>#{cursor}</output> / {horizon}</strong>
            </div>
            <input type="range" min={0} max={horizon} value={cursor} disabled={!horizon}
              aria-label={t("回放位置")} onChange={e => setCursor(Number(e.target.value))} />
            <div className="replay-row replay-milestones" role="group" aria-label={t("关键节点跳转")}>
              <button className="btn" type="button" aria-keyshortcuts="Shift+ArrowLeft"
                disabled={milestones?.previous === undefined}
                onClick={() => milestones?.previous !== undefined && setCursor(milestones.previous)}>
                {t("上一个关键节点")}
              </button>
              <button className="btn" type="button" aria-keyshortcuts="Shift+ArrowRight"
                disabled={milestones?.next === undefined}
                onClick={() => milestones?.next !== undefined && setCursor(milestones.next)}>
                {t("下一个关键节点")}
              </button>
              <small>{!currentStop && milestones?.previous === undefined && milestones?.next === undefined
                ? t("这段历史中没有关键节点。") : "Shift + ← / →"}</small>
            </div>
            <div className="replay-row">
              <button className="btn" type="button" disabled={!cursor} onClick={() => step(-1)}>{t("上一步")}</button>
              <button className="btn" type="button" disabled={cursor >= horizon} onClick={() => step(1)}>{t("下一步")}</button>
            </div>
          </section>
          <section className="replay-event" aria-label={t("当前事件")}>
            {event ? <><h3 aria-live="polite">{t(currentStop?.label ?? "当前事件")} <code>#{event.event_seq}</code></h3>
              <code>{event.event_type}</code>
              <p>{event.stage} · {event.actor}{event.created_at && <> · <time dateTime={event.created_at}>{event.created_at}</time></>}</p>
              {currentStop?.target ? <code>{currentStop.target}</code>
                : publicId(event.payload?.task_id) && <code>{event.payload.task_id}</code>}
            </> : <p>{t("尚无事件记录")}</p>}
          </section>
          <section aria-label={t("已观测任务")}>
            <h3>{t("已观测任务")} · {tasks.length}</h3>
            {!tasks.length ? <p>{t("此时尚无任务状态记录。")}</p> :
              <ul className="replay-tasks">{tasks.map(task => {
                const meta = statusMeta(task.status, task.status === "accepted_with_risk" ? "risk" : null);
                return <li key={task.seq}>
                  <div><strong>{task.id}</strong><small>{task.actor}{task.attempt && <> · {t("第 {n} 次尝试", { n: task.attempt })}</>}</small></div>
                  <span className="badge" data-variant={meta.variant}>{meta.glyph} {meta.label}</span>
                  <button type="button" className="btn" aria-label={t("查看事件 {n}", { n: task.seq })}
                    onClick={() => setCursor(task.seq)}>#{task.seq}</button>
                </li>;
              })}</ul>}
          </section>
          </>}</div>
        </>}
    </dialog>
  );
}
