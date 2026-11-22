import { useRef, useState } from "react";

import type { AmbiguityItem } from "../api/types";
import { localizeTriageReason, MATURITY_LABELS, pendingAmbiguities } from "../lib/labels";
import type { Workbench, WorkbenchState } from "../state/workbench";
import { Badge } from "./ui";

export function IntentCard({ state }: { state: WorkbenchState }) {
  const projection = state.projection;
  const triage = projection?.triage;
  const raw = String(projection?.raw_input || "").trim();
  const hidden = !projection || state.view !== "topology" || (!triage && !raw);
  const maturity = triage?.maturity || "raw_intent";
  const variant = !triage
    ? "muted"
    : maturity === "design_spec_ready"
      ? "ok"
      : maturity === "goal_spec_ready"
        ? "live"
        : "warn";
  const reasons = triage?.reasons || [];
  return (
    <section className="intent-card" id="intent-card" hidden={hidden} aria-label="需求输入与分级">
      <div className="intent-head">
        <span className="intent-label">需求输入</span>
        <Badge variant={variant} id="triage-maturity">
          {triage ? MATURITY_LABELS[maturity] || maturity : projection ? "输入上下文" : "raw_intent"}
        </Badge>
      </div>
      <p className="intent-text" id="intent-text" hidden={!raw}>
        {raw}
      </p>
      <div className="triage-banner" id="triage-banner" data-maturity={maturity} hidden={!triage} aria-live="polite">
        <ul className="triage-reasons" id="triage-reasons">
          {reasons.length ? (
            reasons.map((reason, index) => <li key={index}>{localizeTriageReason(reason)}</li>)
          ) : (
            <li>输入分级已记录</li>
          )}
        </ul>
      </div>
    </section>
  );
}

function ClarifyItem({ amb }: { amb: AmbiguityItem }) {
  const input = useRef<HTMLInputElement>(null);
  const [pressed, setPressed] = useState<string | null>(null);
  const options = amb.options || [];
  return (
    <article className="clarify-item" data-amb-id={amb.id}>
      <div className="clarify-question">
        <Badge variant={amb.blocking ? "warn" : "muted"}>{amb.blocking ? "阻断" : "非阻断"}</Badge>
        <span>{amb.question}</span>
      </div>
      <p className="card-note">{`影响：${amb.impact || "未声明"}`}</p>
      <div className="clarify-options">
        {options.length ? (
          options.map((option) => (
            <button
              type="button"
              className="lane-chip clarify-chip"
              key={option}
              data-amb-id={amb.id}
              data-answer={option}
              aria-pressed={pressed === option ? "true" : "false"}
              onClick={() => {
                if (input.current) input.current.value = option;
                setPressed(option);
              }}
            >
              {option}
            </button>
          ))
        ) : (
          <span className="card-note">无预设选项，请直接填写</span>
        )}
      </div>
      <input
        ref={input}
        className="field clarify-input"
        data-amb-input={amb.id}
        type="text"
        defaultValue={amb.default_assumption || ""}
        placeholder="自定义回答…"
      />
    </article>
  );
}

export function ClarifyPanel({ workbench, state }: { workbench: Workbench; state: WorkbenchState }) {
  const items = useRef<HTMLDivElement>(null);
  const projection = state.projection;
  const pending = projection ? pendingAmbiguities(projection) : [];
  // Answers in progress survive live updates until the question set itself changes.
  const signature = projection
    ? `${projection.run_id}:${state.clarifyGeneration}:${pending.map((amb) => `${amb.id}:${amb.question || ""}`).join("|")}`
    : "";
  const submit = () => {
    const answers: Record<string, string> = {};
    items.current?.querySelectorAll<HTMLInputElement>("[data-amb-input]").forEach((field) => {
      const value = field.value.trim();
      if (value && field.dataset.ambInput) answers[field.dataset.ambInput] = value;
    });
    void workbench.submitClarify(answers);
  };
  return (
    <section className="clarify-panel" id="clarify-panel" hidden={!pending.length} aria-labelledby="clarify-title">
      <div className="card-head">
        <h2 className="card-title" id="clarify-title">
          待澄清问题
        </h2>
        <span className="card-note">回答后生成新的 GoalSpec 版本，旧批准指纹随之失效</span>
      </div>
      <div className="clarify-items" id="clarify-items" ref={items} key={signature}>
        {pending.map((amb) => (
          <ClarifyItem amb={amb} key={amb.id} />
        ))}
      </div>
      <div className="clarify-actions">
        <button type="button" className="btn btn-primary" id="btn-clarify-submit" onClick={submit}>
          提交澄清并推进
        </button>
      </div>
    </section>
  );
}
