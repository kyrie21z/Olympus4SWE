/**
 * Copy for the runtime surface: the API client error messages, the Workbench
 * store (toasts, connection phase text, HITL dialog copy) and the workflow stage
 * model built from a run projection.
 *
 * Chinese keys are translated in `EN`; English keys are translated in `ZH`.
 */
export const EN: Record<string, string> = {
  // connection phase (workbench PHASE_TEXT)
  "正在加载快照…": "Loading snapshot…",
  "找不到运行": "Run not found",
  "正在重连 · 保留上次确认": "Reconnecting · last confirmed state kept",
  "正在提交…": "Submitting…",

  // run command transport
  "操作请求失败：{error}": "Request failed: {error}",
  "状态已同步至最新修订 (r{rev})，请确认后重试":
    "State synced to the latest revision (r{rev}) — review and retry",
  "操作被拒绝：{error}": "Operation rejected: {error}",

  // workspaces
  "加载工作区列表失败": "Failed to load the workspace list",
  "已注册工作区 {name}": "Registered workspace {name}",
  "注册工作区失败": "Failed to register the workspace",
  "已克隆并注册工作区 {name}": "Cloned and registered workspace {name}",
  "克隆工作区失败": "Failed to clone the workspace",
  "已移除工作区注册（本地文件与历史记录均保留）":
    "Workspace unregistered (local files and history are kept)",
  "移除工作区失败：{error}": "Failed to remove the workspace: {error}",
  "移除工作区失败": "Failed to remove the workspace",
  "浏览目录失败": "Failed to browse the directory",

  // artifacts / snapshot loading (api client)
  "读取实机验证产物失败": "Failed to load the product verification artifact",
  "加载运行快照失败": "Failed to load the run snapshot",
  "读取执行产物失败": "Failed to load the artifacts",
  "读取执行产物失败，可重新打开": "Failed to load the artifacts — reopen to retry",

  // approvals
  "GoalSpec v{version} 仍有未决阻断项": "GoalSpec v{version} still has unresolved blockers",
  "请先处理阻断项。若要批准此版本，请填写接受风险的理由。":
    "Resolve the blockers first. To approve this version anyway, state the reason for accepting the risk.",
  "接受风险理由（必填）": "Risk acceptance reason (required)",
  "首期范围已与业务确认，接受未决歧义风险先行冻结规格。":
    "First-phase scope is confirmed with the business; accept the open-ambiguity risk and freeze the spec.",
  "已带风险批准 GoalSpec，进入架构设计": "GoalSpec approved with risk — starting design",
  "已批准 GoalSpec，进入架构设计": "GoalSpec approved — starting design",
  "已批准 DesignSpec，启动任务分解": "DesignSpec approved — starting task decomposition",
  "已批准任务计划，蓝图编排已完成": "Task plan approved — blueprint orchestration complete",
  "已批准任务计划，开始执行": "Task plan approved — starting execution",

  // run commands
  "已暂停新任务派发": "New task dispatch paused",
  "已恢复任务编排调度": "Task orchestration resumed",
  "已升级为 full_sdlc_tdd，启动沙盒代码执行":
    "Promoted to full_sdlc_tdd — starting sandboxed code execution",
  "已从断点重新启动工作流": "Workflow restarted from the checkpoint",
  "已提交修订请求，工作流从需求对齐阶段重新启动":
    "Revision requested — the workflow restarts from alignment",

  // reopen-run dialog
  "追加工单": "Append to run",
  "在已完成的运行上追加后续需求或修复反馈；工作流会从需求对齐阶段重新开始，并保留原有约束与工作区范围。":
    "Append follow-up requirements or repair feedback to a finished run. The workflow restarts from alignment and keeps the original constraints and workspace scope.",
  "追加需求或修复反馈（必填）": "Follow-up requirement or repair feedback (required)",
  "例如：补充导出 Excel 的功能，并保持现有 API 兼容。":
    "e.g. Add an Excel export while keeping the existing API compatible.",
  "已追加工单，工作流从需求对齐阶段重新启动":
    "Appended to the run — the workflow restarts from alignment",

  // banner task actions
  "已将任务 {taskId} 重置为就绪 (READY)": "Task {taskId} reset to READY",
  "向任务 {taskId} 注入人工提示并重试": "Inject a human hint into task {taskId} and retry",
  "提示原文保留在审计事件中，随下次隔离沙盒执行一并注入 Worker 上下文，不直接篡改需求契约。":
    "The hint text is kept in the audit event and injected into the worker context on the next isolated sandbox run; it never rewrites the requirement contract.",
  "注入提示内容": "Hint content",
  "输入要注入的人工提示，例如需修复的边界或重试思路。":
    "Enter the human hint to inject, e.g. the boundary to fix or the retry approach.",
  "已向 {taskId} 注入提示并重新排入就绪队列": "Hint injected into {taskId} — requeued as ready",
  "将任务 {taskId} 改派给人工工程师": "Reassign task {taskId} to a human engineer",
  "改派后人工提交的产物仍须经过相同的程序门禁与 Argus 独立复核才能成为已验收状态。":
    "Artifacts submitted by the human still pass the same programmatic gates and Argus independent review before they can be accepted.",
  "改派备注": "Reassignment note",
  "输入改派备注，例如接手人与需关注的验收要点。":
    "Enter a reassignment note, e.g. the new owner and the acceptance points to watch.",
  "已记录 {taskId} 改派人工并回到就绪态": "{taskId} reassigned to a human and returned to ready",
  "接受风险完成任务 {taskId}": "Accept risk and complete task {taskId}",
  "必须填写明确理由。该决定与 Agent 建议分开存储，追踪矩阵标记为“未验证 · 人工接受”，绝不显示为“已验证”。":
    "An explicit reason is required. The decision is stored separately from the agent recommendation, and the traceability matrix marks it “Unverified · human-accepted”, never “Verified”.",
  "输入接受风险的明确理由，例如影响范围与缓解措施。":
    "Enter the explicit reason for accepting the risk, e.g. blast radius and mitigations.",
  "已记录人工接受风险，{taskId} 标记为未验证 · 人工接受":
    "Risk acceptance recorded — {taskId} marked unverified · human-accepted",

  // clarify / dialog / new run
  "请先回答至少一项澄清问题或填写补充说明":
    "Answer at least one clarification question or add a note first",
  "已保存澄清进度，仍可继续回答剩余阻断项":
    "Clarification progress saved — the remaining blockers can still be answered",
  "已提交澄清并生成新的 GoalSpec 版本":
    "Clarifications submitted — a new GoalSpec version was generated",
  "必须填写理由或提示后才能提交": "A reason or hint is required before submitting",
  "加载工作流或提供方失败": "Failed to load the workflows or providers",
  "请先填写需求原文或导入文件": "Enter the raw requirement text or import a file first",
  "Greenfield 模式必须填写目标目录路径": "Greenfield mode requires a target directory path",
  "请填写现有本地仓库目录路径": "Enter the path of the existing local repository",
  "正在提交中，请稍候": "Submission in progress — hold on",
  "未知错误": "Unknown error",
  "已接收编排 {runId}": "Orchestration accepted: {runId}",

  // workflow stage model (lib/workflowStages.ts)
  "任务编排": "Task orchestration",
  "执行验收": "Execution & verification",
  "对抗审查": "Adversarial review",
  "真实验收": "Surface verification",
  "扩展步骤": "Extended steps",
  "关口": "Gate",
  "校验": "Check",
  "回退{node}": "Fallback → {node}",
  "直达关口": "Direct → gate",
  "直达{node}": "Direct → {node}",
  "回退 {node}": "Fallback → {node}",
  "✕ 中断": "✕ Interrupted",
  "◷ 待开始": "◷ Not started",
};

export const ZH: Record<string, string> = {};
