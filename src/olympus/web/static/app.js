(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const VIEWS = ["topology", "matrix", "waterfall"];
  const TABS = ["overview", "artifacts", "events"];

  const state = {
    runs: [],
    runId: null,
    projection: null,
    view: VIEWS.includes(params.get("view")) ? params.get("view") : "topology",
    selectedTaskId: params.get("task") || null,
    activeTab: TABS.includes(params.get("tab")) ? params.get("tab") : "overview",
    pendingAction: null,
    sse: null,
  };

  const el = {
    runSelector: document.getElementById("run-selector"),
    themeToggle: document.getElementById("theme-toggle"),
    railToggle: document.getElementById("btn-rail-collapse"),
    railLinks: Array.from(document.querySelectorAll(".rail-link[data-view]")),
    panes: Array.from(document.querySelectorAll("[data-pane]")),
    runTitle: document.getElementById("run-title"),
    runKicker: document.getElementById("run-kicker"),
    pipeline: document.getElementById("stage-pipeline"),
    statBand: document.getElementById("stat-band"),
    pauseButton: document.getElementById("btn-toggle-pause"),
    dockOpen: document.getElementById("btn-dock-open"),
    dockClose: document.getElementById("btn-dock-close"),
    resizer: document.getElementById("dock-resizer"),
    dock: document.getElementById("dock"),
    dockTitle: document.getElementById("dock-title"),
    dockStages: document.getElementById("dock-stages"),
    dockSubpane: document.getElementById("dock-subpane"),
    dockTools: document.getElementById("dock-tools"),
    dockTabs: document.getElementById("dock-tabs"),
    dockBody: document.getElementById("dock-body"),
    dockReceipt: document.getElementById("dock-receipt"),
    banner: document.getElementById("hitl-banner"),
    bannerHeadline: document.getElementById("hitl-headline"),
    dag: document.getElementById("dag-svg"),
    lanes: document.getElementById("swimlanes"),
    laneBadge: document.getElementById("lane-badge"),
    frStrip: document.getElementById("fr-strip"),
    matrixBody: document.getElementById("matrix-tbody"),
    waterfall: document.getElementById("waterfall-list"),
    waterfallCount: document.getElementById("waterfall-count"),
    footNote: document.getElementById("foot-note"),
    dialog: document.getElementById("hitl-dialog"),
    dialogTitle: document.getElementById("hitl-dialog-title"),
    dialogDesc: document.getElementById("hitl-dialog-desc"),
    dialogLabel: document.getElementById("hitl-dialog-label"),
    dialogInput: document.getElementById("hitl-dialog-input"),
    toast: document.getElementById("toast-bar"),
  };

  let toastTimer = null;

  function toast(message) {
    el.toast.textContent = message;
    el.toast.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      el.toast.hidden = true;
    }, 3200);
  }

  function statusMeta(status, riskReason) {
    if (riskReason) {
      return { label: "未验证 · 人工接受", variant: "warn", glyph: "⚠" };
    }
    switch (status) {
      case "accepted":
      case "verified":
      case "done":
        return { label: "已验收", variant: "ok", glyph: "✓" };
      case "running":
        return { label: "运行中", variant: "live", glyph: "▶" };
      case "verifying":
        return { label: "待验收", variant: "live", glyph: "▶" };
      case "blocked":
        return { label: "阻塞", variant: "bad", glyph: "✕" };
      case "failed":
        return { label: "失败", variant: "bad", glyph: "✕" };
      case "ready":
        return { label: "就绪", variant: "muted", glyph: "◷" };
      case "cancelled":
        return { label: "已取消", variant: "muted", glyph: "◷" };
      default:
        return { label: "等待", variant: "muted", glyph: "◷" };
    }
  }

  function setTheme(theme) {
    const next = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("olympus-theme", next);
    } catch (error) {
      return;
    }
    el.themeToggle.textContent = next === "dark" ? "☀" : "☾";
    el.themeToggle.setAttribute(
      "aria-label",
      next === "dark" ? "切换到浅色主题" : "切换到深色主题"
    );
  }

  function setView(view) {
    state.view = view;
    el.panes.forEach((pane) => {
      pane.hidden = pane.dataset.pane !== view;
    });
    el.railLinks.forEach((link) => {
      if (link.dataset.view === view) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  }

  function formatTokens(value) {
    const count = Number(value) || 0;
    return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count);
  }

  function renderStage(projection) {
    el.runTitle.textContent = projection.title || "Ariadne 研发编排控制台";
    el.runKicker.textContent = `${projection.mode} · ${projection.status} · r${projection.state_revision}`;

    el.pipeline.innerHTML = (projection.stage_pipeline || [])
      .map((stage, index, all) => {
        const glyph =
          stage.status === "done"
            ? "✓"
            : stage.status === "active"
              ? "▶"
              : stage.status === "blocked"
                ? "✕"
                : "◷";
        const arrow =
          index < all.length - 1 ? '<span class="stage-step-arrow">→</span>' : "";
        return `<span class="stage-step" data-status="${esc(stage.status)}">${glyph} ${esc(
          stage.label
        )}</span>${arrow}`;
      })
      .join("");

    el.statBand.innerHTML = uiStatBand([
      { label: "已验收任务", value: projection.progress.ratio_text },
      {
        label: "LLM 调用",
        value: `${projection.budget.calls_used} / ${projection.budget.max_calls}`,
      },
      { label: "Token 消耗", value: formatTokens(projection.budget.total_tokens) },
      { label: "状态修订", value: `r${projection.state_revision}` },
    ]);

    const paused = projection.status === "paused";
    el.pauseButton.textContent = paused ? "恢复运行" : "暂停全部";
    el.pauseButton.className = paused ? "btn btn-primary" : "btn btn-danger";
    el.footNote.textContent = `${projection.run_id} · 最近事件 #${projection.latest_event_seq}`;
  }

  function renderBanner(projection) {
    const banner = projection.hitl_banner;
    if (!banner) {
      el.banner.hidden = true;
      return;
    }
    el.banner.hidden = false;
    el.banner.dataset.tone = banner.tone || "danger";
    const glyph = banner.tone === "warn" ? "⚠" : banner.tone === "info" ? "◷" : "✕";
    const cleanDetail = String(banner.detail || "")
      .replace(/\s*\n+\s*/g, "，")
      .trim();
    el.bannerHeadline.innerHTML = `<span aria-hidden="true">${glyph}</span><span>${esc(
      banner.headline
    )}<span class="hitl-detail">${esc(cleanDetail)}</span></span>`;
  }

  function renderDag(projection) {
    const nodes = projection.dag.nodes || [];
    const edges = projection.dag.edges || [];
    if (!nodes.length) {
      el.dag.innerHTML =
        '<text class="dag-empty" x="32" y="64">当前运行尚未生成任务 DAG</text>';
      return;
    }

    const byLayer = new Map();
    nodes.forEach((node) => {
      const layer = node.layer || 0;
      if (!byLayer.has(layer)) byLayer.set(layer, []);
      byLayer.get(layer).push(node);
    });
    const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);
    const nodeWidth = 118;
    const nodeHeight = 50;
    const rowGap = 12;
    const columnGap = 168;
    const canvasWidth = Math.max(760, layers.length * columnGap + 40);
    const canvasHeight = 216;
    el.dag.setAttribute("viewBox", `0 0 ${canvasWidth} ${canvasHeight}`);

    const coords = new Map();
    layers.forEach((layer, column) => {
      const columnNodes = byLayer.get(layer);
      const totalHeight = columnNodes.length * nodeHeight + (columnNodes.length - 1) * rowGap;
      const startY = Math.max(12, Math.round((canvasHeight - totalHeight) / 2));
      columnNodes.forEach((node, row) => {
        const x = 24 + column * columnGap;
        const y = startY + row * (nodeHeight + rowGap);
        coords.set(node.task_id, { x, y, node });
      });
    });

    const defs = `<defs>
      <marker id="dag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path class="dag-arrow" d="M0,0 L10,5 L0,10 z"></path></marker>
      <marker id="dag-arrow-blocked" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path class="dag-arrow-blocked" d="M0,0 L10,5 L0,10 z"></path></marker>
    </defs>`;

    const paths = edges
      .map((edge) => {
        const source = coords.get(edge.from);
        const target = coords.get(edge.to);
        if (!source || !target) return "";
        const x1 = source.x + nodeWidth;
        const y1 = source.y + nodeHeight / 2;
        const x2 = target.x;
        const y2 = target.y + nodeHeight / 2;
        const mid = (x1 + x2) / 2;
        const marker = edge.blocked_edge ? "dag-arrow-blocked" : "dag-arrow";
        return `<path class="dag-edge" data-critical="${edge.critical ? "true" : "false"}" data-blocked="${
          edge.blocked_edge ? "true" : "false"
        }" d="M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}" marker-end="url(#${marker})"></path>`;
      })
      .join("");

    const shapes = nodes
      .map((node) => {
        const pos = coords.get(node.task_id);
        if (!pos) return "";
        const meta = statusMeta(node.status, node.accepted_risk_reason);
        const selected = node.task_id === state.selectedTaskId ? "true" : "false";
        const centerX = pos.x + nodeWidth / 2;
        return `<g class="dag-node" data-task-id="${esc(node.task_id)}" data-status="${esc(
          node.status
        )}" data-risk="${node.accepted_risk_reason ? "true" : "false"}" aria-selected="${selected}" tabindex="0" role="button" aria-label="任务 ${esc(
          node.task_id
        )} ${esc(meta.label)}"><rect class="dag-node-box" x="${pos.x}" y="${
          pos.y
        }" width="${nodeWidth}" height="${nodeHeight}" rx="8"></rect><text class="dag-node-title" x="${centerX}" y="${
          pos.y + 23
        }" text-anchor="middle">${esc(node.task_id)}</text><text class="dag-node-meta" x="${centerX}" y="${
          pos.y + 41
        }" text-anchor="middle">${meta.glyph} ${esc(meta.label)}</text></g>`;
      })
      .join("");

    el.dag.innerHTML = defs + paths + shapes;
  }

  function renderLanes(projection) {
    const lanes = projection.swimlanes || [];
    el.laneBadge.textContent = `${lanes.length} 隔离通道`;
    el.lanes.innerHTML = lanes
      .map(
        (lane) => `<div class="lane"><span class="lane-actor">${esc(
          lane.actor
        )}</span><div class="lane-items">${(lane.items || [])
          .map((item) => {
            const meta = statusMeta(item.status, null);
            const pressed = item.task_id && item.task_id === state.selectedTaskId ? "true" : "false";
            return `<button type="button" class="lane-chip" data-task-id="${esc(
              item.task_id || ""
            )}" data-status="${esc(item.status)}" aria-pressed="${pressed}">${meta.glyph} ${esc(
              item.label
            )}</button>`;
          })
          .join("")}</div></div>`
      )
      .join("");
  }

  function renderMatrix(projection) {
    const rows = projection.traceability_matrix || [];
    const counts = { verified: 0, running: 0, failed: 0, risk: 0, pending: 0 };
    rows.forEach((row) => {
      if (row.verification_status === "verified") counts.verified += 1;
      else if (row.verification_status === "running") counts.running += 1;
      else if (row.verification_status === "failed") counts.failed += 1;
      else if (row.verification_status === "unverified_risk_accepted") counts.risk += 1;
      else counts.pending += 1;
    });

    const pills = [];
    if (counts.verified) pills.push(uiBadge(`✓ 已验证 ${counts.verified}`, "ok"));
    if (counts.running) pills.push(uiBadge(`▶ 验证中 ${counts.running}`, "live"));
    if (counts.failed) pills.push(uiBadge(`✕ 失败 ${counts.failed}`, "bad"));
    if (counts.risk) pills.push(uiBadge(`⚠ 人工接受 ${counts.risk}`, "warn"));
    if (counts.pending) pills.push(uiBadge(`◷ 待验证 ${counts.pending}`, "muted"));
    el.frStrip.innerHTML = pills.join("");

    el.matrixBody.innerHTML = rows
      .map((row) => {
        const meta = statusMeta(row.verification_status, null);
        const taskChips = (row.task_ids || [])
          .map(
            (taskId) =>
              `<button type="button" class="lane-chip" data-jump-task="${esc(taskId)}">${esc(
                taskId
              )}</button>`
          )
          .join(" ");
        return `<tr><th scope="row">${esc(row.requirement_id)}</th><td><strong>${esc(
          row.source_ref
        )}</strong> · ${esc(row.description)}</td><td>${
          taskChips || "—"
        }</td><td>${uiBadge(
          `${meta.glyph} ${row.verification_label}`,
          meta.variant
        )} <span class="card-note">${esc(row.evidence_note)}</span></td></tr>`;
      })
      .join("");
  }

  function renderWaterfall(projection) {
    const turns = projection.waterfall_turns || [];
    el.waterfallCount.textContent = `${turns.length} turns`;
    el.waterfall.innerHTML = turns
      .map((turn) => {
        const meta = statusMeta(turn.status, null);
        const offset = Math.min(90, Math.max(0, Number(turn.offset_pct) || 0));
        const width = Math.min(100 - offset, Math.max(8, Number(turn.width_pct) || 8));
        const manifest = turn.manifest_md
          ? `<div class="tool-list"><details class="tool-row"><summary><span class="tool-mark">▤</span><span class="tool-name">MANIFEST.md</span><span class="tool-target">${esc(
              turn.title
            )}</span><span class="tool-duration"></span></summary><pre class="tool-body">${esc(
              turn.manifest_md
            )}</pre></details></div>`
          : "";
        return `<article class="wf-row" data-status="${esc(turn.status)}"><div class="wf-actor"><span>${esc(
          turn.actor
        )}</span>${uiBadge(meta.label, meta.variant)}</div><div class="wf-main"><span class="wf-title">${esc(
          turn.title
        )}</span><span class="wf-summary">${esc(turn.summary)}</span><span class="wf-track"><span class="wf-bar" style="--wf-offset:${offset}%;--wf-fill:${width}%"></span></span><span class="wf-summary">${esc(
          turn.receipt ? turn.receipt.one_line : ""
        )}</span>${manifest}</div><div class="wf-meta"><span>${esc(
          turn.duration_ms
        )} ms</span><span>${esc(turn.steps)} steps</span></div></article>`;
      })
      .join("");
  }

  function currentTask(projection) {
    const nodes = projection.dag.nodes || [];
    return nodes.find((node) => node.task_id === state.selectedTaskId) || nodes[0] || null;
  }

  // Diff lines are tinted by kind only when they carry an actual change; the
  // +++/--- file headers stay neutral context so they never read as edits.
  function diffBlock(patch) {
    if (!patch) {
      return '<pre class="code-block scroller">（该任务暂无 diff.patch 记录）</pre>';
    }
    const rows = String(patch)
      .split("\n")
      .map((line) => {
        let kind = "context";
        if (line.startsWith("+") && !line.startsWith("+++")) kind = "add";
        else if (line.startsWith("-") && !line.startsWith("---")) kind = "del";
        return `<span class="diff-line" data-kind="${kind}">${esc(line)}</span>`;
      })
      .join("\n");
    return `<pre class="code-block code-diff scroller">${rows}</pre>`;
  }

  function renderDockBody(projection, task) {
    if (state.activeTab === "artifacts") {
      const artifacts = task.artifacts || {};
      const verdicts = (artifacts.verdicts || [])
        .map((verdict) => JSON.stringify(verdict, null, 2))
        .join("\n\n");
      return [
        uiCard("diff.patch", diffBlock(artifacts.diff_patch), { note: "隔离沙盒产物" }),
        uiCard(
          "MANIFEST.md",
          `<pre class="code-block scroller">${esc(task.manifest_md || "（暂无 MANIFEST.md）")}</pre>`,
          { note: "完整性清单" }
        ),
        uiCard(
          "Argus 独立复核",
          `<pre class="code-block scroller">${esc(verdicts || "（暂无复核记录）")}</pre>`,
          { note: "冷启动审查" }
        ),
      ].join("");
    }

    if (state.activeTab === "events") {
      const rows = (projection.events || [])
        .slice()
        .reverse()
        .map(
          (event) =>
            `<div class="event-row"><span>#${esc(event.event_seq)}</span><span class="event-type">${esc(
              event.event_type
            )}</span><span class="event-payload">${esc(event.actor)} · r${esc(
              event.state_revision
            )}</span></div>`
        )
        .join("");
      return `<div class="event-log scroller">${rows || uiNotice("info", "暂无事件")}</div>`;
    }

    const checks = (task.acceptance_checks || [])
      .map((check) => `${check.passed ? "✓" : "✕"} ${check.id} · ${check.description}`)
      .join("\n");
    const rows = [
      uiRow("任务目标", esc(task.objective)),
      uiRow("执行者", esc(task.executor), { mono: true }),
      uiRow("写入范围", esc((task.write_scope || []).join(", ") || "（只读）"), { mono: true }),
      uiRow("诊断", `${esc(task.diagnostic_tool)} → ${esc(task.diagnostic_output)}`, { mono: true }),
      uiRow("归因", esc(task.diagnostic_attribution)),
    ].join("");

    return [
      uiCard("执行概览", `<div class="kv">${rows}</div>`),
      task.accepted_risk_reason
        ? uiNotice("warn", `未验证 · 人工接受：${esc(task.accepted_risk_reason)}`)
        : "",
      uiCard("验收检查", `<pre class="code-block scroller">${esc(checks || "（暂无验收检查）")}</pre>`),
    ].join("");
  }

  function renderDock(projection) {
    const task = currentTask(projection);
    if (!task) {
      el.dockTitle.textContent = "任务执行详情";
      el.dockStages.innerHTML = "";
      el.dockSubpane.innerHTML = "";
      el.dockTools.innerHTML = "";
      el.dockBody.innerHTML = '<p class="view-note">暂无选中任务</p>';
      el.dockReceipt.innerHTML = "";
      return;
    }

    state.selectedTaskId = task.task_id;
    const meta = statusMeta(task.status, task.accepted_risk_reason);
    el.dockTitle.textContent = `${task.task_id} · ${task.title}`;
    el.dockStages.innerHTML = [
      uiBadge(`${meta.glyph} ${meta.label}`, meta.variant),
      uiBadge(`尝试 ${task.attempt} / ${task.max_cycles}`, "muted"),
      uiBadge(`审查 ${task.review_cycle} 轮`, "muted"),
      uiBadge(task.executor, "muted"),
    ].join("");

    const live = task.status === "running" || task.status === "verifying";
    el.dockSubpane.innerHTML = subPane(
      `${task.task_id} · 沙盒终端`,
      task.subagent_stream,
      task.terminal_lines,
      { live }
    );
    el.dockTools.innerHTML =
      (task.subagent_stream || []).map((entry) => toolRow(entry)).join("") ||
      uiNotice("info", "该任务尚未执行工具调用");
    el.dockTabs.innerHTML = uiTabs([
      { id: "overview", label: "概览", active: state.activeTab === "overview" },
      { id: "artifacts", label: "产物与审查", active: state.activeTab === "artifacts" },
      { id: "events", label: "事件日志", active: state.activeTab === "events" },
    ]);
    el.dockBody.innerHTML = renderDockBody(projection, task);
    el.dockReceipt.innerHTML = receiptBlock(task.receipt);
  }

  function renderAll(projection) {
    state.projection = projection;
    const nodes = projection.dag.nodes || [];
    if (
      !state.selectedTaskId ||
      !nodes.some((node) => node.task_id === state.selectedTaskId)
    ) {
      const blocked = nodes.find(
        (node) => node.status === "blocked" || node.status === "failed"
      );
      state.selectedTaskId = (blocked || nodes[0] || {}).task_id || null;
    }
    renderStage(projection);
    renderBanner(projection);
    renderDag(projection);
    renderLanes(projection);
    renderMatrix(projection);
    renderWaterfall(projection);
    renderDock(projection);
  }

  function selectTask(taskId) {
    if (!state.projection || !taskId) return;
    state.selectedTaskId = taskId;
    renderDag(state.projection);
    renderLanes(state.projection);
    renderDock(state.projection);
  }

  async function loadRun(runId) {
    state.runId = runId;
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    const data = await response.json();
    if (!data.ok) {
      toast(data.error || "加载运行实例失败");
      return;
    }
    renderAll(data.projection);
    connectStream(runId, data.projection.latest_event_seq || 0);
  }

  function connectStream(runId, afterSeq) {
    if (state.sse) {
      state.sse.close();
      state.sse = null;
    }
    if (typeof EventSource === "undefined") return;
    const source = new EventSource(
      `/api/runs/${encodeURIComponent(runId)}/stream?after_seq=${afterSeq}`
    );
    source.addEventListener("state", async (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (state.projection && payload.state_revision !== state.projection.state_revision) {
          const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
          const data = await response.json();
          if (data.ok) renderAll(data.projection);
        }
      } catch (error) {
        return;
      }
    });
    state.sse = source;
  }

  async function sendCommand(body, successMessage) {
    if (!state.runId || !state.projection) return;
    const payload = Object.assign(
      {
        operation_id: `op-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        expected_state_revision: state.projection.state_revision,
        actor: "zeus",
      },
      body
    );
    const response = await fetch(`/api/runs/${encodeURIComponent(state.runId)}/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!data.ok) {
      toast(`操作被拒绝：${data.error || "未知错误"}`);
      return;
    }
    renderAll(data.projection);
    toast(successMessage);
  }

  function openDialog(config) {
    state.pendingAction = config;
    el.dialogTitle.textContent = config.title;
    el.dialogDesc.textContent = config.description;
    el.dialogLabel.textContent = config.label;
    el.dialogInput.value = config.defaultValue || "";
    el.dialogInput.placeholder = config.placeholder || "";
    el.dialog.showModal();
  }

  function bannerTaskId() {
    const banner = state.projection ? state.projection.hitl_banner : null;
    return (banner && banner.task_id) || state.selectedTaskId;
  }

  function wireEvents() {
    el.runSelector.addEventListener("change", () => {
      state.selectedTaskId = null;
      loadRun(el.runSelector.value);
    });

    el.themeToggle.addEventListener("click", () => {
      const current = document.documentElement.dataset.theme;
      setTheme(current === "dark" ? "light" : "dark");
    });

    el.railToggle.addEventListener("click", () => {
      const collapsed = document.body.dataset.rail === "collapsed";
      document.body.dataset.rail = collapsed ? "expanded" : "collapsed";
      const rail = document.getElementById("rail");
      if (rail) rail.classList.toggle("collapsed", !collapsed);
      el.railToggle.setAttribute("aria-expanded", collapsed ? "true" : "false");
      el.railToggle.textContent = collapsed ? "⇤" : "⇥";
    });

    el.railLinks.forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        setView(link.dataset.view);
      });
    });

    el.dockClose.addEventListener("click", () => {
      document.body.dataset.dock = "closed";
      el.dockOpen.hidden = false;
    });

    el.dockOpen.addEventListener("click", () => {
      document.body.dataset.dock = "open";
      el.dockOpen.hidden = true;
    });

    el.dag.addEventListener("click", (event) => {
      const node = event.target.closest(".dag-node");
      if (node) selectTask(node.dataset.taskId);
    });

    el.dag.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const node = event.target.closest(".dag-node");
      if (!node) return;
      event.preventDefault();
      selectTask(node.dataset.taskId);
    });

    el.lanes.addEventListener("click", (event) => {
      const chip = event.target.closest("[data-task-id]");
      if (chip) selectTask(chip.dataset.taskId);
    });

    el.matrixBody.addEventListener("click", (event) => {
      const chip = event.target.closest("[data-jump-task]");
      if (chip) selectTask(chip.dataset.jumpTask);
    });

    el.dock.addEventListener("click", (event) => {
      const tab = event.target.closest("[data-tab]");
      if (!tab || !state.projection) return;
      state.activeTab = tab.dataset.tab;
      renderDock(state.projection);
    });

    el.pauseButton.addEventListener("click", async () => {
      if (!state.projection) return;
      const action = state.projection.status === "paused" ? "resume" : "pause";
      await sendCommand(
        { action },
        action === "pause" ? "已暂停新任务派发" : "已恢复任务编排调度"
      );
    });

    document.getElementById("btn-banner-inspect").addEventListener("click", () => {
      const taskId = bannerTaskId();
      if (!taskId) return;
      state.activeTab = "overview";
      selectTask(taskId);
      document.body.dataset.dock = "open";
      el.dockOpen.hidden = true;
    });

    document.getElementById("btn-banner-retry").addEventListener("click", async () => {
      const taskId = bannerTaskId();
      if (!taskId) return;
      await sendCommand(
        { action: "retry_task", task_id: taskId, hint: "" },
        `已将任务 ${taskId} 重置为就绪 (READY)`
      );
    });

    document.getElementById("btn-banner-hint").addEventListener("click", () => {
      const taskId = bannerTaskId();
      if (!taskId) return;
      openDialog({
        title: `向任务 ${taskId} 注入人工提示并重试`,
        description:
          "提示原文保留在审计事件中，随下次隔离沙盒执行一并注入 Worker 上下文，不直接篡改需求契约。",
        label: "注入提示内容",
        defaultValue:
          "在 BEGIN IMMEDIATE 事务内同时执行 SELECT WAITLISTED 与 UPDATE status='CONFIRMED'，避免并发取消重复递补。",
        requireNonEmpty: true,
        onConfirm: async (hint) => {
          await sendCommand(
            { action: "retry_task", task_id: taskId, hint },
            `已向 ${taskId} 注入提示并重新排入就绪队列`
          );
        },
      });
    });

    document.getElementById("btn-banner-reassign").addEventListener("click", () => {
      const taskId = bannerTaskId();
      if (!taskId) return;
      openDialog({
        title: `将任务 ${taskId} 改派给人工工程师`,
        description:
          "改派后人工提交的产物仍须经过相同的程序门禁与 Argus 独立复核才能成为已验收状态。",
        label: "改派备注",
        defaultValue: "由值班后端工程师接手修复事务锁边界，提交后走同等验收。",
        requireNonEmpty: true,
        onConfirm: async (note) => {
          await sendCommand(
            { action: "reassign_human", task_id: taskId, note },
            `已记录 ${taskId} 改派人工并回到就绪态`
          );
        },
      });
    });

    document.getElementById("btn-banner-accept-risk").addEventListener("click", () => {
      const taskId = bannerTaskId();
      if (!taskId) return;
      openDialog({
        title: `接受风险完成任务 ${taskId}`,
        description:
          "必须填写明确理由。该决定与 Agent 建议分开存储，追踪矩阵标记为“未验证 · 人工接受”，绝不显示为“已验证”。",
        label: "接受风险理由（必填）",
        defaultValue:
          "首期社团活动仅单管理员人工触发取消，并发取消概率极低，接受已知风险先行放行。",
        requireNonEmpty: true,
        onConfirm: async (reason) => {
          await sendCommand(
            { action: "accept_risk", task_id: taskId, reason },
            `已记录人工接受风险，${taskId} 标记为未验证 · 人工接受`
          );
        },
      });
    });

    // Reset returnValue on close so Esc never leaks a stale "confirm".
    el.dialog.addEventListener("close", async () => {
      const choice = el.dialog.returnValue;
      el.dialog.returnValue = "";
      const pending = state.pendingAction;
      state.pendingAction = null;
      if (choice !== "confirm" || !pending) return;
      const text = el.dialogInput.value.trim();
      if (pending.requireNonEmpty && !text) {
        toast("必须填写理由或提示后才能提交");
        return;
      }
      await pending.onConfirm(text);
    });

    // Light dismiss fallback for browsers without native dialog.closedBy.
    if (!("closedBy" in HTMLDialogElement.prototype)) {
      el.dialog.addEventListener("click", (event) => {
        if (event.target !== el.dialog) return;
        const rect = el.dialog.getBoundingClientRect();
        const inside =
          rect.top <= event.clientY &&
          event.clientY <= rect.top + rect.height &&
          rect.left <= event.clientX &&
          event.clientX <= rect.left + rect.width;
        if (!inside) el.dialog.close();
      });
    }

    // Drag the dock edge; the width is a layout token, never a colour.
    let dragging = false;
    let startX = 0;
    let startWidth = 0;
    const cockpit = document.querySelector(".cockpit");
    el.resizer.addEventListener("pointerdown", (event) => {
      dragging = true;
      startX = event.clientX;
      startWidth = el.dock.getBoundingClientRect().width;
      el.resizer.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    el.resizer.addEventListener("pointermove", (event) => {
      if (!dragging || !cockpit) return;
      const next = Math.min(Math.max(startWidth - (event.clientX - startX), 320), 720);
      cockpit.style.setProperty("--dock-width", `${Math.round(next)}px`);
    });
    el.resizer.addEventListener("pointerup", (event) => {
      dragging = false;
      el.resizer.releasePointerCapture(event.pointerId);
    });
    el.resizer.addEventListener("keydown", (event) => {
      if (!cockpit) return;
      const step = event.key === "ArrowLeft" ? 24 : event.key === "ArrowRight" ? -24 : 0;
      if (!step) return;
      event.preventDefault();
      const width = el.dock.getBoundingClientRect().width + step;
      cockpit.style.setProperty("--dock-width", `${Math.round(Math.min(Math.max(width, 320), 720))}px`);
    });
  }

  async function bootstrap() {
    watchSlots();
    wireEvents();
    setView(state.view);
    setTheme(document.documentElement.dataset.theme || "light");

    let response = await fetch("/api/runs");
    let data = await response.json();
    if (!data.runs || !data.runs.length) {
      response = await fetch("/api/seed-demo", { method: "POST", body: "{}" });
      data = await response.json();
    }
    state.runs = data.runs || [];
    el.runSelector.innerHTML = state.runs
      .map(
        (run) => `<option value="${esc(run.run_id)}">${esc(run.title)} (${esc(run.run_id)})</option>`
      )
      .join("");

    const initialRun =
      params.get("run") || data.default_run_id || (state.runs[0] && state.runs[0].run_id);
    if (initialRun) {
      el.runSelector.value = initialRun;
      await loadRun(initialRun);
    }
  }

  bootstrap();
})();
