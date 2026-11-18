(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const VIEWS = ["topology", "design", "matrix", "waterfall"];
  const TABS = ["overview", "artifacts", "events"];

  const state = {
    runs: [],
    runId: null,
    projection: null,
    view: VIEWS.includes(params.get("view")) ? params.get("view") : "topology",
    selectedTaskId: params.get("task") || null,
    workflowNode: null,
    activeTab: TABS.includes(params.get("tab")) ? params.get("tab") : "overview",
    pendingAction: null,
    optionsLoaded: false,
    workflows: [],
    clarifySig: "",
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
    threadList: document.getElementById("thread-list"),
    newRunButton: document.getElementById("btn-new-run"),
    newRunDialog: document.getElementById("new-run-dialog"),
    newRunTitle: document.getElementById("new-run-title"),
    newRunWorkflow: document.getElementById("new-run-workflow"),
    newRunProvider: document.getElementById("new-run-provider"),
    newRunModel: document.getElementById("new-run-model"),
    newRunInput: document.getElementById("new-run-input"),
    newRunFile: document.getElementById("new-run-file"),
    newRunFileButton: document.getElementById("btn-new-run-file"),
    newRunFileNote: document.getElementById("new-run-file-note"),
    newRunWorkflowMeta: document.getElementById("new-run-workflow-meta"),
    triageBanner: document.getElementById("triage-banner"),
    triageMaturity: document.getElementById("triage-maturity"),
    triageReasons: document.getElementById("triage-reasons"),
    clarifyPanel: document.getElementById("clarify-panel"),
    clarifyItems: document.getElementById("clarify-items"),
    clarifySubmit: document.getElementById("btn-clarify-submit"),
    workflowSvg: document.getElementById("workflow-svg"),
    workflowBadge: document.getElementById("workflow-badge"),
    designBody: document.getElementById("design-body"),
    designBadge: document.getElementById("design-badge"),
    taskActions: document.getElementById("hitl-task-actions"),
    gateActions: document.getElementById("hitl-gate-actions"),
    approveSpec: document.getElementById("btn-approve-spec"),
    approveDesign: document.getElementById("btn-approve-design"),
    approvePlan: document.getElementById("btn-approve-plan"),
  };

  const MATURITY_LABELS = {
    raw_intent: "原始意图",
    goal_spec_ready: "GoalSpec 就绪",
    design_spec_ready: "DesignSpec 就绪",
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
      case "completed":
        return { label: "已完成", variant: "ok", glyph: "✓" };
      case "running":
        return { label: "运行中", variant: "live", glyph: "▶" };
      case "aligning":
        return { label: "对齐中", variant: "live", glyph: "▶" };
      case "designing":
        return { label: "设计中", variant: "live", glyph: "▶" };
      case "planning":
        return { label: "分解中", variant: "live", glyph: "▶" };
      case "awaiting_human":
        return { label: "待人工", variant: "warn", glyph: "⚠" };
      case "created":
        return { label: "已创建", variant: "muted", glyph: "◷" };
      case "verifying":
        return { label: "待验收", variant: "live", glyph: "▶" };
      case "blocked":
        return { label: "阻塞", variant: "bad", glyph: "✕" };
      case "failed":
      case "error":
        return { label: "失败", variant: "bad", glyph: "✕" };
      case "paused":
        return { label: "已暂停", variant: "warn", glyph: "◷" };
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

  function unresolvedAmbiguities(projection) {
    const spec = projection.spec;
    if (!spec) return [];
    return (spec.open_ambiguities || []).filter(
      (amb) => amb && amb.blocking && !amb.answer
    );
  }

  function renderGateActions(projection) {
    const specGate = Boolean(projection.spec && !projection.spec_approved);
    const designGate = Boolean(projection.design && !projection.design_approved);
    const planGate = Boolean(projection.plan && !projection.plan_approved);
    el.approveSpec.hidden = !specGate;
    el.approveDesign.hidden = !designGate;
    el.approvePlan.hidden = !planGate;
    el.gateActions.hidden = !(specGate || designGate || planGate);
    return specGate || designGate || planGate;
  }

  function renderBanner(projection) {
    const banner = projection.hitl_banner;
    const hasGate = renderGateActions(projection);
    el.taskActions.hidden = !(banner && banner.kind === "task_blocked");
    if (!banner && !hasGate) {
      el.banner.hidden = true;
      return;
    }
    el.banner.hidden = false;
    el.banner.dataset.tone = banner ? banner.tone || "danger" : "warn";
    const tone = el.banner.dataset.tone;
    const glyph = tone === "warn" ? "⚠" : tone === "info" ? "◷" : "✕";
    const headline = banner
      ? banner.headline
      : "关口待批准：当前产物已冻结，等待真人确认后继续推进";
    const cleanDetail = String(banner ? banner.detail || "" : "")
      .replace(/\s*\n+\s*/g, "，")
      .trim();
    el.bannerHeadline.innerHTML = `<span aria-hidden="true">${glyph}</span><span>${esc(
      headline
    )}<span class="hitl-detail">${esc(cleanDetail)}</span></span>`;
  }

  function renderTriage(projection) {
    const triage = projection.triage;
    if (!triage) {
      el.triageBanner.hidden = true;
      return;
    }
    const maturity = triage.maturity || "raw_intent";
    el.triageBanner.hidden = false;
    el.triageBanner.dataset.maturity = maturity;
    el.triageMaturity.textContent = MATURITY_LABELS[maturity] || maturity;
    el.triageMaturity.setAttribute(
      "data-variant",
      maturity === "design_spec_ready" ? "ok" : maturity === "goal_spec_ready" ? "live" : "muted"
    );
    const reasons = triage.reasons || [];
    el.triageReasons.innerHTML = reasons.length
      ? reasons.map((reason) => `<li>${esc(reason)}</li>`).join("")
      : "<li>输入分级未附带补充说明</li>";
  }

  function renderClarify(projection) {
    const pending = pendingAmbiguities(projection);
    if (!pending.length) {
      state.clarifySig = "";
      el.clarifyPanel.hidden = true;
      el.clarifyItems.innerHTML = "";
      return;
    }
    const sig = `${projection.run_id}:${pending
      .map((amb) => `${amb.id}:${amb.question || ""}`)
      .join("|")}`;
    el.clarifyPanel.hidden = false;
    if (state.clarifySig === sig && el.clarifyItems.children.length > 0) {
      return;
    }
    state.clarifySig = sig;
    el.clarifyItems.innerHTML = pending
      .map((amb) => {
        const chips = (amb.options || [])
          .map(
            (option) =>
              `<button type="button" class="lane-chip clarify-chip" data-amb-id="${esc(
                amb.id
              )}" data-answer="${esc(option)}" aria-pressed="false">${esc(option)}</button>`
          )
          .join("");
        return `<article class="clarify-item" data-amb-id="${esc(amb.id)}"><div class="clarify-question">${uiBadge(
          amb.blocking ? "阻断" : "非阻断",
          amb.blocking ? "warn" : "muted"
        )}<span>${esc(amb.question)}</span></div><p class="card-note">影响：${esc(
          amb.impact || "未声明"
        )}</p><div class="clarify-options">${
          chips || '<span class="card-note">无预设选项，请直接填写</span>'
        }</div><input class="field clarify-input" data-amb-input="${esc(
          amb.id
        )}" type="text" value="${esc(amb.default_assumption || "")}" placeholder="自定义回答…"></article>`;
      })
      .join("");
  }

  function pendingAmbiguities(projection) {
    const spec = projection.spec;
    if (!spec) return [];
    return (spec.open_ambiguities || []).filter((amb) => amb && !amb.answer);
  }

  function workflowLayers(graph) {
    const names = (graph.nodes || []).map((node) => node.name);
    const known = new Set(names);
    const edges = (graph.edges || []).filter(
      (edge) => known.has(edge.source) && known.has(edge.target)
    );
    const incoming = new Map(names.map((name) => [name, 0]));
    edges.forEach((edge) => incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1));
    const layerOf = new Map();
    const starts = (graph.start || []).filter((name) => known.has(name));
    const queue = (starts.length ? starts : names.filter((name) => !incoming.get(name))).slice();
    if (!queue.length && names.length) queue.push(names[0]);
    queue.forEach((name) => layerOf.set(name, 0));
    let guard = names.length * 8;
    while (queue.length && guard > 0) {
      guard -= 1;
      const name = queue.shift();
      const layer = layerOf.get(name) || 0;
      edges
        .filter((edge) => edge.source === name)
        .forEach((edge) => {
          const next = layer + 1;
          if (!layerOf.has(edge.target) || layerOf.get(edge.target) < next) {
            layerOf.set(edge.target, next);
            queue.push(edge.target);
          }
        });
    }
    names.forEach((name) => {
      if (!layerOf.has(name)) layerOf.set(name, 0);
    });
    return layerOf;
  }

  function renderWorkflow(projection) {
    const workflow = projection.workflow || {};
    const graph = workflow.graph;
    const statuses = workflow.node_statuses || {};
    const name = workflow.name || projection.workflow_name || "workflow";
    if (!graph || !graph.nodes || !graph.nodes.length) {
      el.workflowBadge.textContent = `${name} · 拓扑不可用`;
      el.workflowSvg.innerHTML =
        '<text class="dag-empty" x="32" y="64">当前工作流未提供拓扑描述</text>';
      return;
    }

    const nodeNames = graph.nodes.map((node) => node.name);
    const known = new Set(nodeNames);
    const edges = (graph.edges || []).filter(
      (edge) => known.has(edge.source) && known.has(edge.target)
    );
    const layerOf = workflowLayers(graph);
    const byLayer = new Map();
    graph.nodes.forEach((node) => {
      const layer = layerOf.get(node.name) || 0;
      if (!byLayer.has(layer)) byLayer.set(layer, []);
      byLayer.get(layer).push(node);
    });
    const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);
    const nodeWidth = 118;
    const nodeHeight = 64;
    const rowGap = 14;
    const columnGap = 158;
    const tallest = Math.max(...layers.map((layer) => byLayer.get(layer).length));
    const canvasWidth = Math.max(760, layers.length * columnGap + 48);
    const canvasHeight = Math.max(150, tallest * (nodeHeight + rowGap) + 36);
    el.workflowSvg.setAttribute("viewBox", `0 0 ${canvasWidth} ${canvasHeight}`);

    const coords = new Map();
    layers.forEach((layer, column) => {
      const columnNodes = byLayer.get(layer);
      const totalHeight =
        columnNodes.length * nodeHeight + (columnNodes.length - 1) * rowGap;
      const startY = Math.max(16, Math.round((canvasHeight - totalHeight) / 2));
      columnNodes.forEach((node, row) => {
        coords.set(node.name, {
          x: 24 + column * columnGap,
          y: startY + row * (nodeHeight + rowGap),
          node,
        });
      });
    });

    const defs = `<defs>
      <marker id="wf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path class="dag-arrow" d="M0,0 L10,5 L0,10 z"></path></marker>
      <marker id="wf-arrow-cond" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path class="dag-arrow" d="M0,0 L10,5 L0,10 z"></path></marker>
    </defs>`;

    const paths = edges
      .map((edge) => {
        const source = coords.get(edge.source);
        const target = coords.get(edge.target);
        if (!source || !target) return "";
        const x1 = source.x + nodeWidth;
        const y1 = source.y + nodeHeight / 2;
        const x2 = target.x;
        const y2 = target.y + nodeHeight / 2;
        const mid = (x1 + x2) / 2;
        const marker = edge.conditional ? "wf-arrow-cond" : "wf-arrow";
        return `<path class="dag-edge" data-conditional="${
          edge.conditional ? "true" : "false"
        }" d="M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}" marker-end="url(#${marker})"></path>`;
      })
      .join("");

    const shapes = graph.nodes
      .map((node) => {
        const pos = coords.get(node.name);
        if (!pos) return "";
        const status = statuses[node.name] || "pending";
        const meta = statusMeta(status, null);
        const selected = node.name === state.workflowNode ? "true" : "false";
        const centerX = pos.x + nodeWidth / 2;
        return `<g class="dag-node wf-node" data-workflow-node="${esc(
          node.name
        )}" data-status="${esc(status)}" aria-selected="${selected}" tabindex="0" role="button" aria-label="工作流节点 ${esc(
          node.name
        )} ${esc(meta.label)}"><rect class="dag-node-box" x="${pos.x}" y="${
          pos.y
        }" width="${nodeWidth}" height="${nodeHeight}" rx="8"></rect><text class="dag-node-title" x="${centerX}" y="${
          pos.y + 22
        }" text-anchor="middle">${esc(node.name)}</text><text class="dag-node-kind" x="${centerX}" y="${
          pos.y + 38
        }" text-anchor="middle">${esc(node.kind || "")}</text><text class="dag-node-meta" x="${centerX}" y="${
          pos.y + 55
        }" text-anchor="middle">${meta.glyph} ${esc(meta.label)}</text></g>`;
      })
      .join("");

    el.workflowSvg.innerHTML = defs + paths + shapes;

    const current = workflow.current_node || (workflow.paused && workflow.paused.node);
    el.workflowBadge.textContent = `${name} · ${workflow.topology || "topology"} · stop_after=${
      workflow.stop_after || "-"
    }${current ? ` · 当前 ${current}` : ""}`;
  }

  function requirementList(spec) {
    return (spec.requirements || [])
      .map((req) => {
        const ears = (req.ears_criteria || [])
          .map(
            (criterion) =>
              `<li><span class="card-note">${esc(criterion.pattern || "")}</span> ${esc(
                criterion.condition ? `${criterion.condition} → ${criterion.outcome}` : criterion.outcome
              )}</li>`
          )
          .join("");
        const criteria = (req.acceptance_criteria || [])
          .map((item) => `<li>${esc(item)}</li>`)
          .join("");
        return `<article class="spec-requirement"><div class="spec-requirement-head">${uiBadge(
          req.id,
          "muted"
        )}<span>${esc(req.description)}</span></div><ul class="spec-list">${
          ears || criteria || "<li>（未声明验收标准）</li>"
        }</ul></article>`;
      })
      .join("");
  }

  function renderGoalSpecCard(spec, projection) {
    const surfaces = (spec.observable_surfaces || []).map(esc).join(", ") || "（未声明）";
    const actors = (spec.actors || []).map(esc).join(", ") || "（未声明）";
    const outOfScope = (spec.out_of_scope || []).map((item) => `<li>${esc(item)}</li>`).join("");
    const constraints = (spec.constraints || []).map((item) => `<li>${esc(item)}</li>`).join("");
    const blockers = unresolvedAmbiguities(projection);
    const body = [
      uiRow(
        "目标",
        `${esc(spec.goal)} ${uiBadge(spec.status || "draft", projection.spec_approved ? "ok" : "warn")}`
      ),
      uiRow("交付形态", esc(spec.deliverable_kind || "（未声明）"), { mono: true }),
      uiRow("可观察面", esc(surfaces)),
      uiRow("参与者", esc(actors)),
      uiRow(
        "指纹",
        `${esc(String(spec.content_hash || "").slice(0, 16))}… · v${esc(spec.version || 1)}`,
        { mono: true }
      ),
    ].join("");
    const extras = [
      blockers.length
        ? uiNotice("warn", `仍有 ${blockers.length} 个阻断歧义未回答，批准前需澄清或显式接受风险。`)
        : "",
      uiCard("需求与 EARS 验收", `<div class="spec-requirements">${requirementList(spec)}</div>`),
      uiCard("范围外", `<ul class="spec-list">${outOfScope || "<li>（无）</li>"}</ul>`),
      uiCard("约束", `<ul class="spec-list">${constraints || "<li>（无）</li>"}</ul>`),
    ].join("");
    return uiCard("GoalSpec · 需求冻结", `<div class="kv">${body}</div>${extras}`, {
      note: "目标与验收标准",
    });
  }

  function renderDesignSpecCard(design, projection) {
    const moduleTable = uiTable({
      className: "matrix-table",
      caption: "模块边界与独占写入路径",
      columns: ["模块", "名称", "职责", "关联需求", "目标路径"],
      rows: (design.modules || []).map((module) => [
        `<span class="mono">${esc(module.id)}</span>`,
        esc(module.name),
        esc(module.responsibility),
        `<span class="mono">${esc((module.requirement_refs || []).join(", "))}</span>`,
        `<span class="mono">${esc((module.target_paths || []).join(", "))}</span>`,
      ]),
    });
    const dataModels = (design.data_models || [])
      .map(
        (entity) =>
          `<li><strong>${esc(entity.name)}</strong> <span class="card-note">owner ${
            esc(entity.owner_module_id)
          }</span><br><span class="mono">${esc((entity.fields || []).join(", "))}</span>${
            entity.notes ? ` <span class="card-note">${esc(entity.notes)}</span>` : ""
          }</li>`
      )
      .join("");
    const interfaces = (design.interfaces || [])
      .map(
        (iface) =>
          `<li><strong class="mono">${esc(iface.id)}</strong> ${uiBadge(
            iface.protocol || "http_or_fn",
            "muted"
          )} <span class="mono">${esc(iface.signature)}</span><br><span class="card-note">${esc(
            iface.provider_module_id
          )} → ${esc((iface.consumer_module_ids || []).join(", ") || "external")}</span>${
            (iface.error_cases || []).length
              ? `<br><span class="card-note">错误：${esc((iface.error_cases || []).join("；"))}</span>`
              : ""
          }</li>`
      )
      .join("");
    const risks = (design.risks_and_tradeoffs || [])
      .map((risk) => `<li>${esc(risk)}</li>`)
      .join("");
    const body = [
      uiRow(
        "架构摘要",
        `${esc(design.architecture_summary)} ${uiBadge(
          design.status || "draft",
          projection.design_approved ? "ok" : "warn"
        )}`
      ),
      uiRow("绑定需求", esc(design.spec_id || "（未绑定）"), { mono: true }),
      uiRow(
        "指纹",
        `${esc(String(design.content_hash || "").slice(0, 16))}… · v${esc(design.version || 1)}`,
        { mono: true }
      ),
    ].join("");
    return uiCard(
      "DesignSpec · 白盒设计",
      `<div class="kv">${body}</div>${moduleTable}${uiCard(
        "数据模型",
        `<ul class="spec-list">${dataModels || "<li>（无）</li>"}</ul>`
      )}${uiCard(
        "接口契约",
        `<ul class="spec-list">${interfaces || "<li>（无）</li>"}</ul>`
      )}${uiCard("风险与取舍", `<ul class="spec-list">${risks || "<li>（无）</li>"}</ul>`)}`,
      { note: `${(design.modules || []).length} 个模块` }
    );
  }

  function renderDesign(projection) {
    const spec = projection.spec;
    const design = projection.design;
    el.designBadge.textContent = design
      ? `DesignSpec v${design.version || 1}`
      : spec
        ? `GoalSpec v${spec.version || 1}`
        : "等待规格";
    el.designBadge.setAttribute(
      "data-variant",
      design && projection.design_approved ? "ok" : spec ? "warn" : "muted"
    );
    if (!spec && !design) {
      el.designBody.innerHTML = uiNotice(
        "info",
        "当前运行尚未生成 GoalSpec / DesignSpec。先新建编排提交需求，或批准当前关口后继续推进。"
      );
      return;
    }
    el.designBody.innerHTML = [
      spec ? renderGoalSpecCard(spec, projection) : "",
      design ? renderDesignSpecCard(design, projection) : "",
    ].join("");
  }

  function renderThreads() {
    el.threadList.innerHTML =
      state.runs
        .map((run) => {
          const active = run.run_id === state.runId;
          const meta = statusMeta(run.status, null);
          return `<button type="button" class="thread-item" data-run-id="${esc(
            run.run_id
          )}" aria-pressed="${active ? "true" : "false"}"><span class="thread-item-title">${esc(
            run.title
          )}</span><span class="thread-item-meta"><span class="badge" data-variant="muted">${esc(
            run.workflow_name || "workflow"
          )}</span>${uiBadge(meta.label, meta.variant)}</span></button>`;
        })
        .join("") || '<p class="thread-empty">尚无编排线程</p>';
  }

  async function refreshRuns() {
    const response = await fetch("/api/runs");
    const data = await response.json();
    state.runs = data.runs || [];
    el.runSelector.innerHTML = state.runs
      .map(
        (run) =>
          `<option value="${esc(run.run_id)}">${esc(run.title)} (${esc(run.run_id)})</option>`
      )
      .join("");
    if (state.runId) el.runSelector.value = state.runId;
    renderThreads();
    return data;
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
    const nodeHeight = 62;
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
          pos.y + 22
        }" text-anchor="middle">${esc(node.task_id)}</text><text class="dag-node-kind" x="${centerX}" y="${
          pos.y + 38
        }" text-anchor="middle">${esc(node.role || "engineer")} · ${esc(node.kind || "implement")}</text><text class="dag-node-meta" x="${centerX}" y="${
          pos.y + 54
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
        const moduleChips = (row.design_modules || [])
          .map(
            (module) =>
              `<button type="button" class="lane-chip" data-jump-design="${esc(
                module.id
              )}" title="${esc(module.name || "")}">${esc(module.id)}</button>`
          )
          .join(" ");
        return `<tr><th scope="row">${esc(row.requirement_id)}</th><td><strong>${esc(
          row.source_ref
        )}</strong> · ${esc(row.description)}</td><td>${
          moduleChips || "—"
        }</td><td>${
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
      uiRow("角色 / 类型", `${esc(task.role || "engineer")} / ${esc(task.kind || "implement")}`, {
        mono: true,
      }),
      uiRow("交付物", esc((task.deliverables || []).join(", ") || "（未声明）")),
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

  function renderWorkflowDock(projection, name) {
    const workflow = projection.workflow || {};
    const graph = workflow.graph || { nodes: [], edges: [] };
    const node = (graph.nodes || []).find((item) => item.name === name) || { name };
    const status = (workflow.node_statuses || {})[name] || "pending";
    const meta = statusMeta(status, null);
    const outgoing = (graph.edges || []).filter((edge) => edge.source === name);
    const incoming = (graph.edges || []).filter((edge) => edge.target === name);
    const events = (projection.events || [])
      .filter((event) => event.payload && event.payload.node === name)
      .slice(-12)
      .reverse();

    el.dockTitle.textContent = `${name} · 工作流节点`;
    el.dockStages.innerHTML = [
      uiBadge(`${meta.glyph} ${meta.label}`, meta.variant),
      uiBadge(`类型 ${node.kind || "node"}`, "muted"),
      node.routed ? uiBadge("路由分叉", "warn") : uiBadge("直连节点", "muted"),
      uiBadge(`最大访问 ${node.max_visits || 1}`, "muted"),
      node.on_error ? uiBadge(`失败 → ${node.on_error}`, "bad") : "",
    ].join("");
    el.dockSubpane.innerHTML = "";
    el.dockTools.innerHTML = "";
    el.dockTabs.innerHTML = "";
    const rows = [
      uiRow("节点状态", esc(status), { mono: true }),
      uiRow(
        "入边",
        esc(incoming.map((edge) => edge.source).join(", ") || "（起始节点）"),
        { mono: true }
      ),
      uiRow(
        "出边",
        esc(
          outgoing
            .map((edge) => `${edge.target}${edge.conditional ? "（条件）" : ""}`)
            .join(", ") || "（终点）"
        ),
        { mono: true }
      ),
    ].join("");
    const eventRows = events
      .map(
        (event) =>
          `<div class="event-row"><span>#${esc(event.event_seq)}</span><span class="event-type">${esc(
            event.event_type
          )}</span><span class="event-payload">${esc(event.actor)} · r${esc(
            event.state_revision
          )}</span></div>`
      )
      .join("");
    el.dockBody.innerHTML = [
      uiCard("节点定义", `<div class="kv">${rows}</div>`),
      uiCard(
        "节点事件",
        `<div class="event-log scroller">${
          eventRows || '<p class="view-note">该节点暂无事件</p>'
        }</div>`
      ),
    ].join("");
    el.dockReceipt.innerHTML = "";
  }

  function renderDock(projection) {
    if (state.workflowNode) {
      renderWorkflowDock(projection, state.workflowNode);
      return;
    }
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
      uiBadge(`角色 ${task.role || "engineer"}`, "muted"),
      uiBadge(`类型 ${task.kind || "implement"}`, "muted"),
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
    const runEntry = state.runs.find((run) => run.run_id === projection.run_id);
    if (runEntry) {
      runEntry.status = projection.status;
      runEntry.workflow_name = projection.workflow_name || runEntry.workflow_name;
    }
    renderStage(projection);
    renderBanner(projection);
    renderTriage(projection);
    renderClarify(projection);
    renderWorkflow(projection);
    renderDag(projection);
    renderLanes(projection);
    renderMatrix(projection);
    renderDesign(projection);
    renderWaterfall(projection);
    renderThreads();
    renderDock(projection);
  }

  function selectTask(taskId) {
    if (!state.projection || !taskId) return;
    state.workflowNode = null;
    state.selectedTaskId = taskId;
    renderWorkflow(state.projection);
    renderDag(state.projection);
    renderLanes(state.projection);
    renderDock(state.projection);
  }

  function selectWorkflowNode(name) {
    if (!state.projection || !name) return;
    state.workflowNode = name;
    renderWorkflow(state.projection);
    renderDock(state.projection);
    document.body.dataset.dock = "open";
    el.dockOpen.hidden = true;
  }

  function updateWorkflowMeta() {
    const selected = state.workflows.find(
      (workflow) => workflow.name === el.newRunWorkflow.value
    );
    el.newRunWorkflowMeta.textContent = selected
      ? `${selected.description || "无描述"} · topology: ${selected.topology || "-"} · stop_after: ${
          selected.stop_after || "-"
        }`
      : "";
  }

  async function loadNewRunOptions() {
    if (state.optionsLoaded) return;
    const [workflowsRes, providersRes] = await Promise.all([
      fetch("/api/workflows"),
      fetch("/api/providers"),
    ]);
    const [workflowsData, providersData] = await Promise.all([
      workflowsRes.json(),
      providersRes.json(),
    ]);
    state.workflows = workflowsData.workflows || [];
    el.newRunWorkflow.innerHTML = state.workflows
      .map(
        (workflow) =>
          `<option value="${esc(workflow.name)}">${esc(workflow.name)} · ${esc(
            workflow.topology || "topology"
          )}</option>`
      )
      .join("");
    if (state.workflows.some((workflow) => workflow.name === "product_blueprint")) {
      el.newRunWorkflow.value = "product_blueprint";
    }
    updateWorkflowMeta();

    const providers = providersData.providers || [];
    el.newRunProvider.innerHTML = [
      '<option value="auto">auto · 自动选择已配置提供方</option>',
      ...providers.map(
        (provider) =>
          `<option value="${esc(provider.id)}">${esc(provider.id)} ${
            provider.configured ? "(已配置)" : "(未配置->Mock)"
          }</option>`
      ),
      '<option value="mock">mock · 确定性模拟</option>',
    ].join("");
    state.optionsLoaded = true;
  }

  async function openNewRun() {
    state.workflowNode = null;
    el.newRunDialog.showModal();
    try {
      await loadNewRunOptions();
    } catch (error) {
      toast("加载工作流或提供方失败");
    }
  }

  async function submitNewRun() {
    const rawInput = el.newRunInput.value.trim();
    if (!rawInput) {
      toast("请先填写需求原文或导入文件");
      return;
    }
    const provider = el.newRunProvider.value || "auto";
    const model = el.newRunModel.value.trim();
    const providerConfig = provider === "auto" ? {} : { provider };
    if (model) providerConfig.model = model;
    const payload = {
      raw_input: rawInput,
      title: el.newRunTitle.value.trim() || rawInput.slice(0, 60),
      workflow_name: el.newRunWorkflow.value || "product_blueprint",
      provider_config: providerConfig,
    };
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!data.ok || !data.run_id) {
      toast(`创建失败：${data.error || "未知错误"}`);
      return;
    }
    await refreshRuns();
    el.runSelector.value = data.run_id;
    el.newRunInput.value = "";
    el.newRunTitle.value = "";
    el.newRunModel.value = "";
    el.newRunFileNote.textContent = "支持 .md / .txt / .json / .yaml / .yml";
    state.selectedTaskId = null;
    state.workflowNode = null;
    await loadRun(data.run_id);
    toast(`已创建编排 ${data.run_id} · ${data.workflow || payload.workflow_name}`);
  }

  async function loadRun(runId) {
    state.runId = runId;
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    const data = await response.json();
    if (state.runId !== runId) return;
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
        if (state.runId !== runId) return;
        const payload = JSON.parse(event.data);
        const hasNewEvents =
          Array.isArray(payload.events) &&
          payload.events.some(
            (ev) => (ev.event_seq || 0) > ((state.projection && state.projection.latest_event_seq) || 0)
          );
        if (
          state.projection &&
          (payload.state_revision !== state.projection.state_revision || hasNewEvents)
        ) {
          const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
          const data = await response.json();
          if (state.runId !== runId) return;
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
      state.workflowNode = null;
      loadRun(el.runSelector.value);
    });

    el.threadList.addEventListener("click", (event) => {
      const item = event.target.closest("[data-run-id]");
      if (!item || item.dataset.runId === state.runId) return;
      el.runSelector.value = item.dataset.runId;
      state.selectedTaskId = null;
      state.workflowNode = null;
      loadRun(item.dataset.runId);
    });

    el.newRunButton.addEventListener("click", () => {
      openNewRun();
    });

    el.newRunWorkflow.addEventListener("change", updateWorkflowMeta);

    el.newRunFileButton.addEventListener("click", () => {
      el.newRunFile.click();
    });

    el.newRunFile.addEventListener("change", () => {
      const file = el.newRunFile.files && el.newRunFile.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.addEventListener("load", () => {
        el.newRunInput.value = String(reader.result || "");
        el.newRunFileNote.textContent = `已载入 ${file.name} · ${file.size} bytes`;
      });
      reader.addEventListener("error", () => toast("读取本地文件失败"));
      reader.readAsText(file, "utf-8");
      el.newRunFile.value = "";
    });

    el.newRunDialog.addEventListener("close", async () => {
      const choice = el.newRunDialog.returnValue;
      el.newRunDialog.returnValue = "";
      if (choice !== "create") return;
      await submitNewRun();
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

    el.workflowSvg.addEventListener("click", (event) => {
      const node = event.target.closest("[data-workflow-node]");
      if (node) selectWorkflowNode(node.dataset.workflowNode);
    });

    el.workflowSvg.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const node = event.target.closest("[data-workflow-node]");
      if (!node) return;
      event.preventDefault();
      selectWorkflowNode(node.dataset.workflowNode);
    });

    el.clarifyItems.addEventListener("click", (event) => {
      const chip = event.target.closest(".clarify-chip");
      if (!chip) return;
      const item = chip.closest(".clarify-item");
      const input = item.querySelector("[data-amb-input]");
      if (input) input.value = chip.dataset.answer || "";
      item.querySelectorAll(".clarify-chip").forEach((sibling) => {
        sibling.setAttribute("aria-pressed", sibling === chip ? "true" : "false");
      });
    });

    el.clarifySubmit.addEventListener("click", async () => {
      if (!state.projection) return;
      const answers = {};
      el.clarifyItems.querySelectorAll("[data-amb-input]").forEach((input) => {
        const value = input.value.trim();
        if (value) answers[input.dataset.ambInput] = value;
      });
      const missing = pendingAmbiguities(state.projection).filter(
        (amb) => amb.blocking && !answers[amb.id]
      );
      if (missing.length) {
        toast(`还有 ${missing.length} 个阻断歧义未回答`);
        return;
      }
      if (!Object.keys(answers).length) {
        toast("请先回答至少一个澄清问题");
        return;
      }
      state.clarifySig = "";
      await sendCommand(
        { action: "clarify", answers },
        "已提交澄清并生成新的 GoalSpec 版本"
      );
    });

    el.approveSpec.addEventListener("click", async () => {
      if (!state.projection) return;
      const blockers = unresolvedAmbiguities(state.projection);
      if (blockers.length) {
        openDialog({
          title: "GoalSpec 仍有未决阻断项",
          description:
            "先在上方澄清卡片回答阻断歧义；若必须带风险冻结，请填写接受风险理由，审批会记录为 accept_risk。",
          label: "接受风险理由（必填）",
          defaultValue: "首期范围已与业务确认，接受未决歧义风险先行冻结规格。",
          requireNonEmpty: true,
          onConfirm: async (reason) => {
            await sendCommand(
              { action: "approve_spec", accept_risk: true, reason },
              "已带风险批准 GoalSpec，进入架构设计"
            );
          },
        });
        return;
      }
      await sendCommand(
        { action: "approve_spec" },
        "已批准 GoalSpec，进入架构设计"
      );
    });

    el.approveDesign.addEventListener("click", async () => {
      await sendCommand(
        { action: "approve_design" },
        "已批准 DesignSpec，启动任务分解"
      );
    });

    el.approvePlan.addEventListener("click", async () => {
      await sendCommand({ action: "approve_plan" }, "已批准任务计划，开始执行");
    });

    el.lanes.addEventListener("click", (event) => {
      const chip = event.target.closest("[data-task-id]");
      if (chip) selectTask(chip.dataset.taskId);
    });

    el.matrixBody.addEventListener("click", (event) => {
      const designChip = event.target.closest("[data-jump-design]");
      if (designChip) {
        setView("design");
        return;
      }
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

    let data = await refreshRuns();
    if (!state.runs.length) {
      await fetch("/api/seed-demo", { method: "POST", body: "{}" });
      data = await refreshRuns();
    }

    const initialRun =
      params.get("run") || data.default_run_id || (state.runs[0] && state.runs[0].run_id);
    if (initialRun) {
      el.runSelector.value = initialRun;
      await loadRun(initialRun);
    }
  }

  bootstrap();
})();
