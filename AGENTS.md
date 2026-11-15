# AGENTS.md — Olympus4SWE

Olympus4SWE 是一个以第一性原理驱动的 Agent-Native 全生命周期软件工程系统（Python 3.13 + `uv`）。

## 仓库地图

- 架构与设计原则：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 本地工程参考资料（未入库）：`docs/references/oiloil-curated/INDEX.md`
- 核心源码：`src/olympus/`
  - `core/` — 强类型契约、SHA-256 内容指纹、DAG 与 `write_scope` 校验、SQLite 事件与状态存储
  - `providers/` — 轻量 LLM 接口（OpenAI-Compatible / Anthropic / Mock）
  - `tools/` — 绝对路径防呆 ACI 原语与输出截断引导
  - `harness/` — 透明 Append-Only Loop、`git worktree` 沙盒、Trace 落盘
  - `stages/` — Stage 1 `Align`（需求对齐冻结）与 Stage 2 `Orchestrate & Verify`（调度与独立验收）
  - `web/` — M4 `Ariadne` 事件溯源只读投影、SSE 实时流与幂等人工控制 Web 控制台
  - `cli/` — CLI 交互入口
- 测试与评测：`tests/unit/`、`tests/evals/`

## 开发与验证约定

- 依赖与命令统一通过 `uv run` 执行（`uv run pytest`、`uv run ruff check .`）。
- 确定性控制面（状态迁移、DAG 校验、预算、锁、审批绑定）严禁交由 LLM 直接改写。
- `providers/` 为独立可插拔的模型兼容层（参考 Pi Agent `pi-ai` 等架构），核心调度与 Stage 严禁绑定/强耦合特定厂商（如百炼）。
- 面向可移植性：`core/`、`tools/`、`stages/` 不强耦合第三方编排框架，方便部分模块迁回姊妹仓 `SWE-Olympus`。
- 开发分工：新增模块/功能动手前，先派 Paseo `CDS`/`QDS` subagent 在 GitHub 检索同类实现以借鉴或复用；确定性编码与信息检索优先由 `CDS`/`QDS` 执行，高级模型专注方案交流与细化。
- Git 交付工作流：除仓库初始化与微不足道的单行拼写微调外，新需求、功能增强与缺陷修复统一遵循 `GitHub Issue → Branch / git worktree → PR（Closes #<id>）` 闭环合入 `main`，严禁直接向 `main` 堆叠未跟踪的新特性改动。
