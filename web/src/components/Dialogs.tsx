import { useEffect, useRef, useState, type ReactNode } from "react";

import { WORKFLOW_SELECT_LABELS, WORKFLOW_SUMMARY } from "../lib/labels";
import type { NewRunForm, Workbench, WorkbenchState } from "../state/workbench";

const FILE_NOTE = "支持 .md / .txt / .json / .yaml / .yml";

/**
 * Thin wrapper over a native `<dialog closedby="any">`: the platform owns modality, focus trap,
 * Esc and light dismiss; React only renders the form and receives the `close` return value.
 */
export function NativeDialog({
  id,
  labelledBy,
  onClose,
  lightDismissFallback = false,
  children,
}: {
  id: string;
  labelledBy: string;
  onClose: (returnValue: string) => void;
  /** Emulate `closedby="any"` light dismiss where the browser lacks `HTMLDialogElement.closedBy`. */
  lightDismissFallback?: boolean;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return undefined;
    const handleClose = () => {
      const choice = dialog.returnValue;
      // Reset so Esc or light dismiss never replays a stale confirm.
      dialog.returnValue = "";
      onCloseRef.current(choice);
    };
    const handleClick = (event: MouseEvent) => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const inside =
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width;
      if (!inside) dialog.close();
    };
    const needsFallback = lightDismissFallback && !("closedBy" in HTMLDialogElement.prototype);
    dialog.addEventListener("close", handleClose);
    if (needsFallback) dialog.addEventListener("click", handleClick);
    return () => {
      dialog.removeEventListener("close", handleClose);
      if (needsFallback) dialog.removeEventListener("click", handleClick);
    };
  }, [lightDismissFallback]);

  return (
    <dialog ref={ref} id={id} aria-labelledby={labelledBy} {...{ closedby: "any" }}>
      {children}
    </dialog>
  );
}

const EMPTY_FORM: NewRunForm = {
  demoId: "",
  title: "",
  workflow: "",
  provider: "auto",
  model: "",
  workerDriver: "",
  reviewerDriver: "",
  monitorModel: "",
  input: "",
};

export function NewRunDialog({ workbench, state }: { workbench: Workbench; state: WorkbenchState }) {
  const [form, setForm] = useState<NewRunForm>(EMPTY_FORM);
  const [fileNote, setFileNote] = useState(FILE_NOTE);
  const fileInput = useRef<HTMLInputElement>(null);
  const formRef = useRef(form);
  formRef.current = form;
  const options = state.options;
  const workflows = options?.workflows || [];
  const demos = options?.demos || [];
  const providers = options?.providers || [];

  // Default to the planning-only blueprint once the catalog arrives.
  useEffect(() => {
    if (!options) return;
    setForm((current) =>
      current.workflow
        ? current
        : {
            ...current,
            workflow: options.workflows.some((wf) => wf.name === "product_blueprint")
              ? "product_blueprint"
              : options.workflows[0]?.name || "",
          },
    );
  }, [options]);

  const set = (patch: Partial<NewRunForm>) => setForm((current) => ({ ...current, ...patch }));
  const selectedWorkflow = workflows.find((workflow) => workflow.name === form.workflow);
  const workflowMeta = selectedWorkflow
    ? `${WORKFLOW_SUMMARY[selectedWorkflow.name] || selectedWorkflow.description || "无描述"} · 编写代码：${
        selectedWorkflow.stop_after === "execute"
          ? "是（沙盒执行 + 验收）"
          : `否（止步于 ${selectedWorkflow.stop_after || "plan"}）`
      }`
    : "";
  const demo = demos.find((item) => item.id === form.demoId);
  const demoNote = demo
    ? [
        demo.workspace_summary || "独立隔离 git 工作区",
        (demo.constraints || []).length ? `初始约束：${demo.constraints.join("；")}` : "",
        (demo.clarification_hints || []).length ? `建议澄清参考：${demo.clarification_hints.join(" / ")}` : "",
      ]
        .filter(Boolean)
        .join(" · ")
    : "";

  const applyDemo = (demoId: string) => {
    const preset = demos.find((item) => item.id === demoId);
    if (!preset) {
      set({ demoId });
      return;
    }
    set({
      demoId,
      title: preset.title || "",
      input: preset.raw_input || "",
      ...(preset.default_workflow && workflows.some((wf) => wf.name === preset.default_workflow)
        ? { workflow: preset.default_workflow }
        : {}),
    });
  };

  const onClose = async (choice: string) => {
    if (choice !== "create") return;
    const created = await workbench.submitNewRun(formRef.current);
    if (!created) return;
    setForm((current) => ({ ...current, demoId: "", input: "", title: "", model: "" }));
    setFileNote(FILE_NOTE);
  };

  const onFile = () => {
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      set({ input: String(reader.result || "") });
      setFileNote(`已载入 ${file.name} · ${file.size} bytes`);
    });
    reader.addEventListener("error", () => workbench.toast("读取本地文件失败"));
    reader.readAsText(file, "utf-8");
    if (fileInput.current) fileInput.current.value = "";
  };

  return (
    <NativeDialog id="new-run-dialog" labelledBy="new-run-dialog-title" onClose={(choice) => void onClose(choice)}>
      <form method="dialog" id="new-run-form">
        <h2 className="dialog-title" id="new-run-dialog-title">
          新建编排
        </h2>
        <p className="dialog-desc">
          从一句话需求或既有文档启动一条受控编排线程。每个关口都会停下等待真人批准，不会自动越过。
        </p>
        <div className="form-field">
          <label className="dialog-label" htmlFor="new-run-demo">
            Benchmark 预设样本（可选）
          </label>
          <select id="new-run-demo" className="field" value={form.demoId} onChange={(event) => applyDemo(event.target.value)}>
            {options ? <option value="">空白需求（自由输入 · 主仓库工作区）</option> : null}
            {demos.map((item) => (
              <option value={item.id} key={item.id}>
                {item.label || item.title || item.id}
              </option>
            ))}
          </select>
        </div>
        <p className="dialog-desc" id="new-run-demo-note" hidden={!demoNote}>
          {demoNote}
        </p>
        <div className="form-grid">
          <div className="form-field">
            <label className="dialog-label" htmlFor="new-run-title">
              标题（可选）
            </label>
            <input
              id="new-run-title"
              className="field"
              type="text"
              placeholder="例如：校园候补递补加固"
              value={form.title}
              onChange={(event) => set({ title: event.target.value })}
            />
          </div>
          <div className="form-field">
            <label className="dialog-label" htmlFor="new-run-workflow">
              工作流
            </label>
            <select
              id="new-run-workflow"
              className="field"
              value={form.workflow}
              onChange={(event) => set({ workflow: event.target.value })}
            >
              {workflows.map((workflow) => (
                <option value={workflow.name} key={workflow.name}>
                  {WORKFLOW_SELECT_LABELS[workflow.name] || `${workflow.name} · ${workflow.topology || "topology"}`}
                </option>
              ))}
            </select>
          </div>
          <div className="form-field">
            <label className="dialog-label" htmlFor="new-run-provider">
              模型提供方
            </label>
            <select
              id="new-run-provider"
              className="field"
              value={form.provider}
              onChange={(event) => set({ provider: event.target.value })}
            >
              {options ? (
                <>
                  <option value="auto">auto · 自动选择已配置提供方</option>
                  {providers.map((provider) => (
                    <option value={provider.id} key={provider.id}>
                      {`${provider.id} ${provider.configured ? "(已配置)" : "(未配置->Mock)"}`}
                    </option>
                  ))}
                  <option value="mock">mock · 确定性模拟</option>
                </>
              ) : null}
            </select>
          </div>
          <div className="form-field">
            <label className="dialog-label" htmlFor="new-run-model">
              模型（可选）
            </label>
            <input
              id="new-run-model"
              className="field"
              type="text"
              placeholder="留空使用提供方默认模型"
              value={form.model}
              onChange={(event) => set({ model: event.target.value })}
            />
          </div>
          <div className="form-field">
            <label className="dialog-label" htmlFor="new-run-worker-driver">
              执行驱动（可选）
            </label>
            <input
              id="new-run-worker-driver"
              className="field"
              type="text"
              placeholder="留空使用环境配置"
              value={form.workerDriver}
              onChange={(event) => set({ workerDriver: event.target.value })}
            />
          </div>
          <div className="form-field">
            <label className="dialog-label" htmlFor="new-run-reviewer-driver">
              审查驱动（可选）
            </label>
            <input
              id="new-run-reviewer-driver"
              className="field"
              type="text"
              placeholder="留空使用环境配置"
              value={form.reviewerDriver}
              onChange={(event) => set({ reviewerDriver: event.target.value })}
            />
          </div>
          <div className="form-field">
            <label className="dialog-label" htmlFor="new-run-monitor-model">
              监控模型（可选）
            </label>
            <input
              id="new-run-monitor-model"
              className="field"
              type="text"
              placeholder="留空使用环境配置"
              value={form.monitorModel}
              onChange={(event) => set({ monitorModel: event.target.value })}
            />
          </div>
        </div>
        <p className="dialog-desc" id="new-run-workflow-meta">
          {workflowMeta}
        </p>
        <label className="dialog-label" htmlFor="new-run-input">
          需求原文
        </label>
        <textarea
          id="new-run-input"
          className="field"
          rows={8}
          placeholder="描述目标、可观察面、范围与约束…"
          value={form.input}
          onChange={(event) => set({ input: event.target.value })}
        />
        <div className="dialog-file-row">
          <button type="button" className="btn" id="btn-new-run-file" onClick={() => fileInput.current?.click()}>
            导入本地文件
          </button>
          <input
            type="file"
            id="new-run-file"
            accept=".md,.txt,.json,.yaml,.yml"
            hidden
            ref={fileInput}
            onChange={onFile}
          />
          <span className="card-note" id="new-run-file-note">
            {fileNote}
          </span>
        </div>
        <div className="dialog-actions">
          <button type="submit" className="btn btn-quiet" value="cancel">
            取消
          </button>
          <button type="submit" className="btn btn-primary" id="new-run-submit" value="create" disabled={state.submitting}>
            创建并启动
          </button>
        </div>
      </form>
    </NativeDialog>
  );
}

export function HitlDialog({ workbench, state }: { workbench: Workbench; state: WorkbenchState }) {
  const config = state.dialog;
  return (
    <NativeDialog
      id="hitl-dialog"
      labelledBy="hitl-dialog-title"
      lightDismissFallback
      onClose={(choice) => void workbench.closeDialog(choice)}
    >
      <form method="dialog" id="hitl-dialog-form">
        <h2 className="dialog-title" id="hitl-dialog-title">
          {config?.title || "人工关口操作"}
        </h2>
        <p className="dialog-desc" id="hitl-dialog-desc">
          {config?.description || ""}
        </p>
        <label className="dialog-label" htmlFor="hitl-dialog-input" id="hitl-dialog-label">
          {config?.label || "补充说明"}
        </label>
        <textarea id="hitl-dialog-input" className="field" rows={3} placeholder="请输入提示或风险接受理由…" />
        <div className="dialog-actions">
          <button type="submit" className="btn btn-quiet" value="cancel">
            取消
          </button>
          <button type="submit" className="btn btn-primary" id="hitl-dialog-submit" value="confirm">
            确认提交
          </button>
        </div>
      </form>
    </NativeDialog>
  );
}

export function Toast({ state }: { state: WorkbenchState }) {
  return (
    <div className="toast" id="toast-bar" role="status" aria-live="polite" hidden={!state.toast.visible}>
      {state.toast.message}
    </div>
  );
}
