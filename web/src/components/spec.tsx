import type { DesignSpec, GoalSpec, RunProjection } from "../api/types";
import { unresolvedAmbiguities } from "../lib/labels";
import { Badge, Card, DataTable, KeyValues, Mono, Notice, Row } from "./ui";

type SchemaField = NonNullable<NonNullable<DesignSpec["interfaces"]>[number]["input_schema"]>[number];

function formatSchemaFields(fields: SchemaField[] | undefined): string {
  return (fields || [])
    .map(
      (field) =>
        `${field.name}: ${field.type_expr}${field.required === false ? "?" : ""}${
          field.constraints ? ` (${field.constraints})` : ""
        }`,
    )
    .join(", ");
}

export function DesignTradeoff({ choice, cost }: { choice: string; cost: string }) {
  return (
    <>
      <span>选择：{choice}</span>
      <br />
      <span>代价：{cost}</span>
    </>
  );
}

export function DesignStructure({ design }: { design: DesignSpec }) {
  const contracts = design.interfaces || [];
  return (
    <>
      <DataTable
        label="模块图"
        caption="模块图 · 模块边界与独占写入路径"
        columns={["模块", "名称", "职责", "关联需求", "目标路径", "依赖模块"]}
        rows={(design.modules || []).map((module) => {
          const providers = new Set(
            contracts
              .filter((iface) => (iface.consumer_module_ids || []).includes(module.id))
              .map((iface) => iface.provider_module_id),
          );
          return {
            key: module.id,
            cells: [
              <Mono>{module.id}</Mono>,
              module.name,
              module.responsibility,
              <Mono>{(module.requirement_refs || []).join(", ")}</Mono>,
              <Mono>{(module.target_paths || []).join(", ")}</Mono>,
              <Mono>{[...providers].join(", ") || "—"}</Mono>,
            ],
          };
        })}
      />
      <Card title="数据模型">
        <ul className="spec-list">
          {(design.data_models || []).length ? (
            (design.data_models || []).map((entity) => (
              <li key={entity.name}>
                <strong>{entity.name}</strong> <span className="card-note">owner {entity.owner_module_id}</span>
                <br />
                <span className="mono">{(entity.fields || []).join(", ")}</span>
                {entity.notes ? (
                  <>
                    {" "}
                    <span className="card-note">{entity.notes}</span>
                  </>
                ) : null}
              </li>
            ))
          ) : (
            <li>（无）</li>
          )}
        </ul>
      </Card>
      <Card title="接口契约">
        <ul className="spec-list">
          {contracts.length ? (
            contracts.map((iface) => {
              const inText = formatSchemaFields(iface.input_schema);
              const outText = formatSchemaFields(iface.output_schema);
              const refsText = (iface.data_entity_refs || []).join(", ");
              const errors = iface.error_cases || [];
              return (
                <li key={iface.id}>
                  <strong className="mono">{iface.id}</strong> <Badge>{iface.protocol || "http_or_fn"}</Badge>{" "}
                  <span className="mono">{iface.signature}</span>
                  <br />
                  <span className="card-note">
                    {iface.provider_module_id} → {(iface.consumer_module_ids || []).join(", ") || "external"}
                  </span>
                  {inText ? (
                    <>
                      <br />
                      <span className="card-note">
                        入参：<span className="mono">{inText}</span>
                      </span>
                    </>
                  ) : null}
                  {outText ? (
                    <>
                      <br />
                      <span className="card-note">
                        出参：<span className="mono">{outText}</span>
                      </span>
                    </>
                  ) : null}
                  {refsText ? (
                    <>
                      <br />
                      <span className="card-note">
                        实体：<span className="mono">{refsText}</span>
                      </span>
                    </>
                  ) : null}
                  {errors.length ? (
                    <>
                      <br />
                      <span className="card-note">错误：{errors.join("；")}</span>
                    </>
                  ) : null}
                </li>
              );
            })
          ) : (
            <li>（无）</li>
          )}
        </ul>
      </Card>
    </>
  );
}

function RequirementList({ spec }: { spec: GoalSpec }) {
  return (
    <>
      {(spec.requirements || []).map((req) => {
        const ears = req.ears_criteria || [];
        const criteria = req.acceptance_criteria || [];
        return (
          <article className="spec-requirement" key={req.id}>
            <div className="spec-requirement-head">
              <Badge>{req.id}</Badge>
              <span>{req.description}</span>
            </div>
            <ul className="spec-list">
              {ears.length
                ? ears.map((criterion, index) => (
                    <li key={criterion.id || index}>
                      <span className="card-note">{criterion.pattern || ""}</span>{" "}
                      {criterion.condition ? `${criterion.condition} → ${criterion.outcome}` : criterion.outcome}
                    </li>
                  ))
                : criteria.length
                  ? criteria.map((item, index) => <li key={index}>{item}</li>)
                  : <li>（未声明验收标准）</li>}
            </ul>
          </article>
        );
      })}
    </>
  );
}

function ListOrNone({ items }: { items: string[] | undefined }) {
  return (
    <ul className="spec-list">
      {(items || []).length ? (items || []).map((item, index) => <li key={index}>{item}</li>) : <li>（无）</li>}
    </ul>
  );
}

export function GoalSpecCard({ spec, projection }: { spec: GoalSpec; projection: RunProjection }) {
  const blockers = unresolvedAmbiguities(projection.spec);
  return (
    <Card title="GoalSpec · 需求冻结" note="目标与验收标准" id="run-spec-detail">
      <KeyValues>
        <Row label="目标">
          {spec.goal} <Badge variant={projection.spec_approved ? "ok" : "warn"}>{spec.status || "draft"}</Badge>
        </Row>
        <Row label="交付形态" mono>
          {spec.deliverable_kind || "（未声明）"}
        </Row>
        <Row label="可观察面">{(spec.observable_surfaces || []).join(", ") || "（未声明）"}</Row>
        <Row label="参与者">{(spec.actors || []).join(", ") || "（未声明）"}</Row>
        <Row label="指纹" mono>
          {`${String(spec.content_hash || "").slice(0, 16)}… · v${spec.version || 1}`}
        </Row>
      </KeyValues>
      {blockers.length ? (
        <Notice level="warn">{`仍有 ${blockers.length} 个阻断歧义未回答，批准前需澄清或显式接受风险。`}</Notice>
      ) : null}
      <Card title="需求与 EARS 验收">
        <div className="spec-requirements">
          <RequirementList spec={spec} />
        </div>
      </Card>
      <Card title="范围外">
        <ListOrNone items={spec.out_of_scope} />
      </Card>
      <Card title="约束">
        <ListOrNone items={spec.constraints} />
      </Card>
    </Card>
  );
}

const IMPACTS: Record<string, string> = { existing_system: "现有系统", deployment: "部署", rollback: "回滚" };

export function DesignSpecCard({ design, projection }: { design: DesignSpec; projection: RunProjection }) {
  const invariants = design.shared_invariants || [];
  const tradeoffs = design.design_tradeoffs || [];
  const risks = design.risks_and_tradeoffs || [];
  const strategy = design.execution_strategy;
  const strategyBadge =
    strategy === "contract_first_parallel" ? (
      <Badge variant="ok">契约先行并发</Badge>
    ) : strategy === "serial_handoff" ? (
      <Badge variant="warn">串行顺序接力</Badge>
    ) : null;
  const contractPaths = (design.shared_contract_paths || []).join(", ");
  return (
    <Card title="DesignSpec · 白盒设计" note={`${(design.modules || []).length} 个模块`} id="run-design-detail">
      <KeyValues>
        <Row label="架构摘要">
          {design.architecture_summary}{" "}
          <Badge variant={projection.design_approved ? "ok" : "warn"}>{design.status || "draft"}</Badge>
        </Row>
        {strategyBadge ? (
          <Row label="执行策略">
            {strategyBadge}
            {design.strategy_rationale ? (
              <>
                {" "}
                <span className="card-note">{design.strategy_rationale}</span>
              </>
            ) : null}
          </Row>
        ) : null}
        {contractPaths ? (
          <Row label="共享契约域" mono>
            {contractPaths} <Badge>W0 写 · W1 只读</Badge>
          </Row>
        ) : null}
        <Row label="绑定需求" mono>
          {design.spec_id || "（未绑定）"}
        </Row>
        <Row label="指纹" mono>
          {`${String(design.content_hash || "").slice(0, 16)}… · v${design.version || 1}`}
        </Row>
      </KeyValues>
      {invariants.length ? (
        <DataTable
          caption="全局不变量（防跨模块隐性漂移）"
          columns={["编号", "类别", "全局规则", "适用范围 / 提示"]}
          rows={invariants.map((inv) => ({
            key: inv.id,
            cells: [
              <Mono>{inv.id}</Mono>,
              <Badge>{inv.category || "rule"}</Badge>,
              inv.rule,
              <>
                {(inv.applies_to || []).join(", ") || "全局"}
                {inv.enforcement_hint ? (
                  <>
                    {" · "}
                    <span className="card-note">{inv.enforcement_hint}</span>
                  </>
                ) : null}
              </>,
            ],
          }))}
        />
      ) : (
        <Card title="全局不变量">
          <p className="view-note">v1 规格未声明结构化全局不变量（DesignSpec v2 新建架构将强制校验）。</p>
        </Card>
      )}
      <DesignStructure design={design} />
      <Card title="风险与取舍">
        <ul className="spec-list">
          {tradeoffs.length || risks.length ? (
            <>
              {tradeoffs.map(({ impact, choice, cost }, index) => (
                <li key={`t${index}`}>
                  <strong>{IMPACTS[impact]}</strong>
                  <br />
                  <DesignTradeoff choice={choice} cost={cost} />
                </li>
              ))}
              {risks.map((risk, index) => (
                <li key={`r${index}`}>{risk}</li>
              ))}
            </>
          ) : (
            <li>（无）</li>
          )}
        </ul>
      </Card>
    </Card>
  );
}
