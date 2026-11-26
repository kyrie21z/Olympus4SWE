# PR 审查标准

<!-- hydrogenjr: olympus4swe-review-v1 -->

本文件规定 `Olympus4SWE` 的 PR 在什么条件下可以合并、什么条件下阻断，以及常驻审查代理 `Hydrogen`（`gemini-4-hydrogen-medium`）与自动合并守护进程如何协作。人工审查采用同一套标准。修改本文件须走 PR，并由 `@kyrie21z` 人工审批合并。

验收以作者在 PR 描述与关联 Issue 中写明的预期行为为依据，重点核查架构边界、契约一致性、兼容性、测试覆盖与视觉证据。预期行为缺失或前后矛盾时，要求作者补充具体条件。

## 一、三种结论

每次审查只给出以下三种结论之一：

| 结论 | 含义 | 后续动作 |
| --- | --- | --- |
| ✅ **Approve** | 必选 CI 全绿，代码与证据支持实现符合预期，且无任何阻断项 | `Hydrogen` 批准当前提交；守护进程核对合并条件并按规则处理 |
| ❌ **Request changes** | 确认存在第二节任一阻断项，或 CI 报告了可归因于本 PR 改动的失败 | 指出具体编号、文件位置、预期与实际行为；作者修改并更新后重审 |
| 🟡 **待核验** | 预期描述、CI 结果、截图或复现证据仍有缺口，暂无法确认无回归 | 先核查现有可获取信息，再向作者列明需要补充的具体证据 |

## 二、阻断项（命中任一条均不得合并）

**流程与范围（B1–B4）**

- **B1.** 未关联 GitHub Issue（缺少 `Closes #<id>`），或 PR 描述缺少“改了什么 / 怎么测 / 风险”中任一项，或只写“测过了”“没问题”这类空话。
- **B2.** 目标分支不是 `main`，与最新 `main` 存在合并冲突，或 PR 仍处于 Draft 状态（未转为 `Ready for review`）。
- **B3.** 单个 PR 混杂多件无关改动（标题一句话无法概括全部变更，或同时修改多个无依赖关系的模块）；或手写代码超过约 800 行（不含生成的 `src/olympus/web/static/`、`src/olympus/web/openapi.json`、`web/src/api/schema.d.ts`、锁文件与验证截图）却未说明不拆分的理由。
- **B4.** **HW3 外部贡献者后端红线**：外部贡献者未经 `@kyrie21z` 预先批准的契约 Issue，擅自修改 `src/olympus/**/*.py` 或 `src/olympus/web/openapi.json`。确需新增后端只读字段或接口时，必须先提交契约 Issue 并 `@kyrie21z`，待后端契约合入后再基于更新后的 `web/src/api/schema.d.ts` 继续开发。

**架构与契约（B5–B8）**

- **B5.** 违反 `AGENTS.md` 中的 6 条架构不变量（由 `tests/unit/test_architecture_invariants.py` 机械校核）：`src/olympus/core/` 纯净度、`src/olympus/providers/` 厂商 SDK 隔离、核心/运行时与展示层（`web/`、`cli/`）隔离、`tests/evals/ → src/olympus/` 单向依赖、前端类型与契约纪律，以及控制面与 LLM 面分离。
- **B6.** 破坏确定性控制面保护：允许 LLM 输出直接修改状态机流转、DAG 校验结果、`write_scope` / `DiffGate` 门禁、单调预算账本或哈希绑定审批。
- **B7.** 违反前端工程纪律：`web/src/` 下出现未加类型的 `.js`/`.jsx` 文件；引入冗余或重型 UI 组件库全家桶（如 AntD、MUI、ECharts 或多套并存框架）及外部监控栈（如 Prometheus、Grafana、Langfuse、Docker）；或在生产视图中捏造、硬编码假指标或假回放数据（所有数据须来自真实 API，缺失字段用诚实空态呈现）。
- **B8.** 违反设计系统规范（由 `tests/unit/test_design_system.py` 校核）：在 `web/src/styles/tokens.css` 之外的样式、TS/TSX 或 HTML 中写入裸 `#hex`、`rgb()`、`rgba()`、`hsl()`、`hsla()` 颜色字面量；新增颜色或图表 Token 未集中写在 `web/src/styles/tokens.css` 且未同时适配亮色与暗色主题；或私自删改、绕过设计系统测试。

**构建与卫生（B9–B11）**

- **B9.** 构建产物与契约未同步：修改 `web/` 后未同步生成 `src/olympus/web/static/`（须通过 `ariadne-source-sha256` 校验，且禁止提交 `.map` 文件）；修改后端 API Schema 后未同步更新 `src/olympus/web/openapi.json`、`web/src/api/schema.d.ts` 与 `src/olympus/web/static/`。
- **B10.** 提交了本机绝对路径、密钥、`.env`、调试残留文件，或未说明理由即删除他人负责模块的代码或文件。
- **B11.** 在 `pyproject.toml` 或 `web/package.json` 中新增依赖，却未在 PR 描述中说明引入理由与包体积影响，或未同步更新锁文件（`uv.lock` / `web/pnpm-lock.yaml`）。

**确定性 CI 与证据（B12–B14）**

- **B12.** 必选 CI 检查（当前为 `check`、`frontend`，待浏览器 CI 落地后包含 `browser`）未全绿，或存在可归因于本 PR 的失败。所有确定性静态检查与测试均由 CI 执行，`Hydrogen` 不在审查沙箱中重复运行。若检查因 CI 环境波动未完成，先重试或以 `🟡 待核验` 说明缺口。
- **B13.** 涉及 UI 或视觉交互变化的 PR，未在描述中附上真实运行截图（建议同时附亮色与暗色主题）。
- **B14.** 新增核心逻辑或前端交互缺少对应的单元测试或 `tests/unit/test_web_*_browser.py` 浏览器测试，且未在 PR 描述中给出明确理由。

以下三类问题不作为硬阻断（标记为 `[Nit]` 建议项），但需作者回应或按需改进：命名或代码风格与周边模块不一致；引入新领域概念却未更新 `GLOSSARY.md`；重要架构取舍未写入 `docs/adr/`。作者给出合理解释即可 Approve。

## 三、确定性 CI 与分级核验

GitHub Actions（`.github/workflows/ci.yml`）承担全部确定性检查。`Hydrogen` 消费 CI 状态与失败日志片段，不在沙箱内手动重跑测试套件：

| CI 任务 | 状态 | 执行内容 |
| --- | --- | --- |
| `check` | 必选 | `uv sync --locked`、`uv run --no-sync ruff check .`、`uv run --no-sync mypy src`、`uv run --no-sync pytest -q`（包含 `test_architecture_invariants.py`、`test_design_system.py`、`test_web_openapi_contract.py` 等） |
| `frontend` | 必选 | `pnpm --dir web install --frozen-lockfile`、拦截 `web/src/` 下 `.js`/`.jsx`、`pnpm --dir web generate:api` 与 `schema.d.ts` diff 校验、`pnpm --dir web typecheck && pnpm --dir web build` 与 `src/olympus/web/static` diff 校验 |
| `browser` | 落地后必选（见 `#155`） | 运行关键 Playwright 浏览器场景（`tests/unit/test_web_*_browser.py`） |

`Hydrogen` 按改动涉及的最高风险路径在评论中标注审查级别：

| 级别 | 涉及路径 | 审查重点 |
| --- | --- | --- |
| `docs` | 仅 `*.md`、`docs/` | B1–B3、B10、与现有契约和命令的一致性 |
| `web` | `web/`、`src/olympus/web/static/`、`tests/unit/test_web_*_browser.py` | B1–B4、B7–B14、交互体验、真实 API 消费、亮暗主题截图与浏览器测试 |
| `evals` | `tests/evals/` | B1–B5、B10–B12、B14、评测独立性与可复现性 |
| `backend` | `src/olympus/`、`.github/`、`pyproject.toml`、`tests/unit/` | B1–B14 全部阻断项、控制面确定性与维护者路径核对 |

## 四、定向检查与维护者专属审批路径

### 定向检查触发点

遇到以下改动时，审查评论中必须写出具体核对证据：

- `web/src/styles/tokens.css` 或主题样式：核对亮色与暗色主题下的对比度与可读性，确认无裸色值与无回归截图。
- `src/olympus/web/openapi.json` 或 `web/src/api/schema.d.ts`：核对新增/变更字段的前后端兼容性、空态处理以及 HITL 命令的幂等与乐观锁语义。
- 删除或移动文件与模块：检索所有调用方、导出符号、资源引用与测试，确认无悬空引用且符合 B10。
- 新增依赖（`pyproject.toml` 或 `web/package.json`）：核对 PR 描述中的引入理由、包体积影响与锁文件同步（B11），以及是否符合 B7 依赖克制要求。

### 维护者专属审批路径（守护进程绝不自动合并）

触及以下任一路径的 PR，涉及仓库治理规则、构建配置或 Platform Core 确定性内核，**守护进程一律不自动合并**，必须由 `@kyrie21z` 人工审批合并（`Hydrogen` 仍照常发表技术审查结论）：

- 治理、规范与构建配置：
  - `AGENTS.md`
  - `CONTRIBUTING.md`
  - `docs/REVIEW.md`
  - `.github/`
  - `pyproject.toml`
- Platform Core：
  - `src/olympus/core/`
  - `src/olympus/harness/`
  - `src/olympus/stages/`
  - `src/olympus/session/`
  - `src/olympus/graph/engine.py`
  - `src/olympus/graph/loader.py`
  - `src/olympus/graph/progress.py`
  - `src/olympus/web/server.py`
  - `src/olympus/web/schemas.py`
  - `src/olympus/web/projection.py`
  - `src/olympus/web/supervisor.py`
  - `src/olympus/web/openapi.py`

## 五、自动合并策略与 Hydrogen 工作方式

**审查代理**：`Hydrogen`（模型：`gemini-4-hydrogen-medium`），在隔离沙箱中只读审查 PR。

**触发与合并规则**

- **Draft PR**：跳过审查与自动合并；转为 `Ready for review` 后触发。
- **`@kyrie21z` 常规 PR**：非 Draft、必选 CI 全绿、与最新 `main` 无冲突，且**未触及第四节维护者专属审批路径**时，由守护进程直接执行 **Squash and merge**，无需人工批准。
- **外部贡献者 PR**：非 Draft、必选 CI 全绿、`Hydrogen` 对当前 `HEAD` 提交发表 `✅ Approve`、与最新 `main` 无冲突，且**未触及第四节维护者专属审批路径**时，由守护进程自动执行 **Squash and merge**。
- **触及维护者专属审批路径的 PR**：无论作者是 `@kyrie21z` 还是外部贡献者，守护进程绝不自动合并，等待 `@kyrie21z` 人工合并。

**Hydrogen 每次审查的步骤**

1. 读取 PR 描述与关联 Issue，检查 B1、B2、B3 与 B4（确认外部贡献者是否越界触碰后端代码或 `openapi.json`）。
2. 读取必选 CI 检查结果（`check`、`frontend`，及落地后的 `browser`），核对 B12。**不在本地重复运行确定性测试**；仅在某项检查失败时截取对应失败日志片段定位原因。
3. 从目标分支（`main`）读取本文件与 `AGENTS.md`，对照 PR 与 Issue 中声明的预期审读 diff。跳过 `src/olympus/web/static/` 压缩构建产物正文（仅核对同步状态与无 `.map` 文件），沿调用方与契约边界逐项核对 B5–B14 与第四节定向检查。重审时聚焦本轮新提交及受影响调用方，确认先前阻断项已消除且最新 `HEAD` 的 CI 全绿。
4. 按固定格式发表审查结论并执行对应动作：
   - **Approve**：批准当前已审查提交；由守护进程在满足全部合并条件时执行 Squash and merge。
   - **Request changes**：评论列明阻断项与修改方向，不合并。
   - **待核验**：评论列明缺失的预期描述、CI 状态或视觉/测试证据，等待作者补充。

**权限边界**

- 不向 PR 分支推送提交，不关闭 PR，不修改 Issue；修改建议统一写在审查评论中，由作者自行修改。
- 自动合并只由守护进程在规则全部满足时执行；维护者专属审批路径由 `@kyrie21z` 人工合并。
- 不配置也不读取任何密钥或敏感凭证。
- 无法从代码、CI 与截图确认的事项明确写“不确定”，不猜测。

**审查评论格式**

```text
结论：Approve / Request changes / 待核验
级别：docs / web / evals / backend

CI 状态：（列出 check / frontend / browser 的实际结论；若失败附关键报错摘要）

检查结果：（说明已核对的预期、定向检查证据与架构/契约检查结论）

阻断项：
- B? 文件:行 —— 问题描述与修复方向

建议项（不阻断）：
- [Nit] ...

未验证：（如无头环境未实机交互的视觉细节、待作者确认的边界条件等）
```

## 六、作者要做什么

- 按 `.github/pull_request_template.md` 填写 PR 描述，包含 `Closes #<id>` 与“改了什么 / 怎么测 / 风险”；“怎么测”重点写明手动验证路径与观察到的实际结果。
- 推送前在本地跑通对应验证命令，确保 CI 一次通过：
  ```bash
  # 后端或通用检查
  uv run ruff check .
  uv run mypy src
  uv run pytest -q

  # 改动 web/ 或 API 契约时额外执行
  pnpm --dir web generate:api
  pnpm --dir web typecheck
  pnpm --dir web build
  ```
- 涉及 UI 或视觉交互变化时，在 PR 描述中附真实运行截图（建议含亮色与暗色主题）。
- 对 `Hydrogen` 的意见有不同看法时，直接在评论中给出代码依据或复现证据；若补充了新的验收预期，同步更新 PR 描述以便重审。
