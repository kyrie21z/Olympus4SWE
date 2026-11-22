import { Fragment, type ReactNode } from "react";

import type { SubagentStep, TurnReceipt } from "../api/types";
import type { Variant } from "../lib/labels";

export function Badge({ variant = "muted", children, id }: { variant?: Variant | string; children: ReactNode; id?: string }) {
  return (
    <span className="badge" data-variant={variant} id={id}>
      {children}
    </span>
  );
}

export function Card({
  title,
  note,
  id,
  children,
}: {
  title?: string;
  note?: string;
  id?: string;
  children?: ReactNode;
}) {
  return (
    <section className="card" id={id}>
      {title ? (
        <div className="card-head">
          <h3 className="card-title">{title}</h3>
          {note ? <span className="card-note">{note}</span> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function Notice({
  level = "info",
  children,
  action,
}: {
  level?: "info" | "warn" | "bad" | "ok";
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="notice" data-level={level} role="status">
      <div>{children}</div>
      {action}
    </div>
  );
}

export interface StatItem {
  label: string;
  value: string;
}

export function StatCells({ items }: { items: StatItem[] }) {
  return (
    <>
      {items.map((item) => (
        <div className="stat-cell" key={item.label}>
          <span className="stat-label">{item.label}</span>
          <span className="stat-value">{item.value}</span>
        </div>
      ))}
    </>
  );
}

export function Row({ label, mono, children }: { label: string; mono?: boolean; children: ReactNode }) {
  return (
    <div className="kv-row">
      <span className="kv-key">{label}</span>
      <span className={mono ? "kv-value kv-value-mono" : "kv-value"}>{children}</span>
    </div>
  );
}

export function KeyValues({ children }: { children: ReactNode }) {
  return <div className="kv">{children}</div>;
}

/** Region-wrapped table whose first cell in each row is the row header. */
export function DataTable({
  label,
  caption,
  columns,
  rows,
  className = "matrix-table",
}: {
  label?: string;
  caption?: string;
  columns: string[];
  rows: { key: string; cells: ReactNode[] }[];
  className?: string;
}) {
  return (
    <div className="table-wrapper scroller" tabIndex={0} role="region" aria-label={label || "数据表"}>
      <table className={className}>
        {caption ? <caption>{caption}</caption> : null}
        <thead>
          <tr>
            {columns.map((column) => (
              <th scope="col" key={column}>
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ key, cells }) => (
            <tr key={key}>
              <th scope="row">{cells[0]}</th>
              {cells.slice(1).map((cell, index) => (
                <td key={index}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Mono({ children }: { children: ReactNode }) {
  return <span className="mono">{children}</span>;
}

export function ToolRow({ tool }: { tool: SubagentStep }) {
  const status = tool.status || "ok";
  const mark = status === "ok" ? "✓" : status === "denied" ? "⊘" : "✕";
  const duration = typeof tool.duration_ms === "number" && tool.duration_ms > 0 ? `${tool.duration_ms} ms` : "";
  const preview = (tool as SubagentStep & { output_preview?: string }).output_preview || tool.target || "（无输出）";
  return (
    <details className="tool-row" data-status={status}>
      <summary>
        <span className="tool-mark" aria-hidden="true">
          {mark}
        </span>
        <span className="tool-name">{tool.tool}</span>
        <span className="tool-target">{tool.target || ""}</span>
        <span className="tool-duration">{duration}</span>
      </summary>
      <pre className="tool-body">{preview}</pre>
    </details>
  );
}

function lineTone(text: string): string {
  if (text.indexOf("$ ") === 0) return "muted";
  if (/failed|error|✕|⚠/i.test(text)) return "bad";
  if (/passed|通过|✓/i.test(text)) return "ok";
  return "muted";
}

export function SubPane({
  title,
  tools,
  tailLines,
  live,
}: {
  title: string;
  tools: unknown[];
  tailLines: string[];
  live: boolean;
}) {
  const lines = (tailLines || []).slice(-4).map(String);
  return (
    <section className="subpane" data-live={live ? "true" : "false"}>
      <div className="subpane-head">
        <span className="subpane-title">{title}</span>
        <span className="subpane-live">
          <span className="subpane-dot" aria-hidden="true" />
          {`${live ? "live" : "idle"} · ${(tools || []).length} tools`}
        </span>
      </div>
      <pre className="subpane-body">
        {lines.length ? (
          lines.map((text, index) => (
            <span className="subpane-line" data-tone={lineTone(text)} key={index}>
              {text}
            </span>
          ))
        ) : (
          <span className="subpane-line" data-tone="muted">
            （暂无终端输出）
          </span>
        )}
      </pre>
    </section>
  );
}

export function ReceiptBlock({ receipt }: { receipt: TurnReceipt | null | undefined }) {
  if (!receipt) return null;
  const llm = receipt.llm || { calls: 0, input_tokens: 0, output_tokens: 0 };
  const artifacts = receipt.artifacts || { diff_added: 0, diff_removed: 0, checks_passed: 0, checks_total: 0 };
  return (
    <div className="receipt" aria-label="Turn receipt">
      <span className="receipt-line">
        <span className="receipt-key">回执</span> {receipt.turn_id || ""}
      </span>
      <span className="receipt-line">{receipt.one_line || ""}</span>
      <span className="receipt-line">
        {`模型 ${llm.calls || 0} 次 · ${llm.input_tokens || 0} in / ${llm.output_tokens || 0} out`}
      </span>
      <span className="receipt-line">
        {`补丁 +${artifacts.diff_added || 0}/−${artifacts.diff_removed || 0} · 门禁 ${artifacts.checks_passed || 0}/${artifacts.checks_total || 0}`}
      </span>
    </div>
  );
}

/** Diff lines are tinted only when they carry a change; +++/--- headers stay neutral. */
export function DiffBlock({ patch }: { patch: string | null | undefined }) {
  if (!patch) return <pre className="code-block scroller">（该任务暂无 diff.patch 记录）</pre>;
  const lines = String(patch).split("\n");
  return (
    <pre className="code-block code-diff scroller">
      {lines.map((line, index) => {
        let kind = "context";
        if (line.startsWith("+") && !line.startsWith("+++")) kind = "add";
        else if (line.startsWith("-") && !line.startsWith("---")) kind = "del";
        return (
          <Fragment key={index}>
            <span className="diff-line" data-kind={kind}>
              {line}
            </span>
            {index < lines.length - 1 ? "\n" : null}
          </Fragment>
        );
      })}
    </pre>
  );
}
