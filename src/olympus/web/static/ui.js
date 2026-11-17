"use strict";

/* Olympus4SWE · Ariadne UI primitives.
 *
 * Pure string builders. Nothing here names a colour, a shadow or a type size:
 * every visual decision belongs to style.css and tokens.css.
 */

function esc(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function uiBadge(text, variant) {
  return `<span class="badge" data-variant="${esc(variant || "muted")}">${esc(text)}</span>`;
}

function uiButton(label, opts) {
  const options = opts || {};
  const classes = ["btn"];
  if (options.variant === "accent") classes.push("btn-accent");
  else if (options.variant === "danger") classes.push("btn-danger");
  else if (options.variant === "quiet") classes.push("btn-quiet");
  const attrs = [`type="button"`, `class="${classes.join(" ")}"`];
  if (options.id) attrs.push(`id="${esc(options.id)}"`);
  if (options.action) attrs.push(`data-action="${esc(options.action)}"`);
  if (options.taskId) attrs.push(`data-task-id="${esc(options.taskId)}"`);
  if (options.title) attrs.push(`title="${esc(options.title)}"`);
  if (options.disabled) attrs.push("disabled");
  return `<button ${attrs.join(" ")}>${esc(label)}</button>`;
}

function uiCard(title, body, opts) {
  const options = opts || {};
  const head = title
    ? `<div class="card-head"><h3 class="card-title">${esc(title)}</h3>${
        options.note ? `<span class="card-note">${esc(options.note)}</span>` : ""
      }</div>`
    : "";
  const idAttr = options.id ? ` id="${esc(options.id)}"` : "";
  return `<section class="card"${idAttr}>${head}${body}</section>`;
}

function uiTable(spec) {
  const options = spec || {};
  const head = (options.columns || [])
    .map((column) => `<th scope="col">${esc(column)}</th>`)
    .join("");
  const rows = (options.rows || [])
    .map((row) => {
      const cells = row.cells || row;
      const rowHead = `<th scope="row">${cells[0]}</th>`;
      const rest = cells
        .slice(1)
        .map((cell) => `<td>${cell}</td>`)
        .join("");
      return `<tr>${rowHead}${rest}</tr>`;
    })
    .join("");
  return `<div class="table-wrapper scroller" tabindex="0" role="region" aria-label="${esc(
    options.label || "数据表"
  )}"><table class="${esc(options.className || "matrix-table")}">${
    options.caption ? `<caption>${esc(options.caption)}</caption>` : ""
  }<thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function uiTabs(tabs) {
  const items = (tabs || [])
    .map(
      (tab) =>
        `<button type="button" role="tab" class="tab" data-tab="${esc(tab.id)}" aria-selected="${
          tab.active ? "true" : "false"
        }">${esc(tab.label)}</button>`
    )
    .join("");
  return `<div class="tabs" role="tablist" aria-label="任务详情视图切换">${items}</div>`;
}

function uiNotice(level, html, action) {
  return `<div class="notice" data-level="${esc(level || "info")}" role="status"><div>${html}</div>${
    action || ""
  }</div>`;
}

function uiStatBand(items) {
  return (items || [])
    .map(
      (item) =>
        `<div class="stat-cell"><span class="stat-label">${esc(item.label)}</span><span class="stat-value">${esc(
          item.value
        )}</span></div>`
    )
    .join("");
}

function uiRow(label, value, opts) {
  const options = opts || {};
  const valueClass = options.mono ? "kv-value kv-value-mono" : "kv-value";
  return `<div class="kv-row"><span class="kv-key">${esc(label)}</span><span class="${valueClass}">${value}</span></div>`;
}

function toolRow(tool) {
  const entry = tool || {};
  const status = entry.status || "ok";
  const mark = status === "ok" ? "✓" : status === "denied" ? "⊘" : "✕";
  const duration =
    typeof entry.duration_ms === "number" && entry.duration_ms > 0 ? `${entry.duration_ms} ms` : "";
  const preview = entry.output_preview || entry.target || "（无输出）";
  return `<details class="tool-row" data-status="${esc(status)}"><summary><span class="tool-mark" aria-hidden="true">${mark}</span><span class="tool-name">${esc(
    entry.tool
  )}</span><span class="tool-target">${esc(entry.target || "")}</span><span class="tool-duration">${esc(
    duration
  )}</span></summary><pre class="tool-body">${esc(preview)}</pre></details>`;
}

function subPane(title, tools, tailLines, opts) {
  const options = opts || {};
  const lines = (tailLines || []).slice(-4);
  const body = lines
    .map((line) => {
      const text = String(line);
      let tone = "muted";
      if (text.indexOf("$ ") === 0) tone = "muted";
      else if (/failed|error|✕|⚠/i.test(text)) tone = "bad";
      else if (/passed|通过|✓/i.test(text)) tone = "ok";
      return `<span class="subpane-line" data-tone="${tone}">${esc(text)}</span>`;
    })
    .join("");
  const count = (tools || []).length;
  const state = options.live ? "live" : "idle";
  const collapsed = options.open === false ? " hidden" : "";
  return `<section class="subpane" data-live="${options.live ? "true" : "false"}"${collapsed}><div class="subpane-head"><span class="subpane-title">${esc(
    title
  )}</span><span class="subpane-live"><span class="subpane-dot" aria-hidden="true"></span>${esc(
    state
  )} · ${count} tools</span></div><pre class="subpane-body">${
    body || '<span class="subpane-line" data-tone="muted">（暂无终端输出）</span>'
  }</pre></section>`;
}

function receiptBlock(receipt) {
  if (!receipt) return "";
  const llm = receipt.llm || {};
  const artifacts = receipt.artifacts || {};
  const lines = [
    `<span class="receipt-line"><span class="receipt-key">回执</span> ${esc(receipt.turn_id || "")}</span>`,
    `<span class="receipt-line">${esc(receipt.one_line || "")}</span>`,
    `<span class="receipt-line">模型 ${esc(llm.calls || 0)} 次 · ${esc(
      llm.input_tokens || 0
    )} in / ${esc(llm.output_tokens || 0)} out</span>`,
    `<span class="receipt-line">补丁 +${esc(artifacts.diff_added || 0)}/−${esc(
      artifacts.diff_removed || 0
    )} · 门禁 ${esc(artifacts.checks_passed || 0)}/${esc(artifacts.checks_total || 0)}</span>`,
  ];
  return `<div class="receipt" aria-label="Turn receipt">${lines.join("")}</div>`;
}

function stampSlots(root) {
  if (!root || typeof root.querySelectorAll !== "function") return;
  const slots = {
    button: "button",
    select: "select",
    textarea: "field",
    input: "field",
    dialog: "dialog",
    table: "table",
  };
  const stamp = (node) => {
    if (!node || !node.tagName) return;
    const slot = slots[node.tagName.toLowerCase()];
    if (slot && !node.dataset.slot) node.dataset.slot = slot;
  };
  stamp(root);
  root.querySelectorAll("button, select, textarea, input, dialog, table").forEach(stamp);
}

function watchSlots() {
  if (typeof MutationObserver === "undefined") return;
  stampSlots(document.body);
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      record.addedNodes.forEach((node) => {
        if (node.nodeType === 1) stampSlots(node);
      });
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}
