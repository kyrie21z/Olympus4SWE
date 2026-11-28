import type { RunEvent } from "../api/types";
import { publicId } from "./evidenceId";
import { positiveAttempt, taskFailureLabel } from "./taskIncidents";

type Milestone = { seq: number; label: string; target?: string };

const approvalLabels = new Map([
  ["goal_spec", "需求规格已批准"], ["design_spec", "设计规格已批准"],
  ["task_plan", "任务计划已批准"], ["final_acceptance", "最终验收已批准"],
]);

function milestoneLabel(event: RunEvent): string | undefined {
  const p = event.payload ?? {};
  const failure = taskFailureLabel(event);
  if (failure) return failure;
  switch (event.event_type) {
    case "approval.recorded":
      if (p.decision === "approved") return approvalLabels.get(String(p.target)) ?? "审批通过";
      if (p.decision === "rejected") return "审批被退回";
      if (p.decision === "accept_risk") return "审批接受风险";
      return;
    case "check.command_completed": return p.passed === false ? "验收检查未通过" : undefined;
    case "review.verdict_recorded": return p.passed === true ? "独立审查通过" : undefined;
    case "task.retried_by_human": return "人工已请求重试";
    case "task.reassigned_human": return "已改派人工";
    case "task.accepted_with_risk": return "人工接受风险放行";
    case "task.leased": return (positiveAttempt(p.attempt) ?? 0) > 1 ? "重试已开始" : undefined;
    case "task.accepted": return "任务已验收";
    case "run.revision_requested": return "人工要求修订";
    case "run.reopened": return "运行已重新打开";
    case "post_dag.repair_rejected": return "交付修复被退回";
    case "delivery.checks_completed": return "交付检查已结束";
    case "product.verified":
      if (p.passed === true) return "产品实机验证通过";
      if (p.passed === false) return "产品实机验证未通过";
      return;
    case "post_dag.repro_rechecked": {
      const checked = positiveAttempt(p.checked_count);
      const failed = p.still_failing_count;
      if (!checked || typeof failed !== "number" || !Number.isInteger(failed) || failed < 0 || failed > checked) return;
      return failed === 0 ? "交付复验通过" : "交付复验未通过";
    }
  }
}

/** Index only classified events from the frozen run prefix. */
export function indexMilestones(events: RunEvent[]) {
  const stops: Milestone[] = [];
  for (const event of events) {
    const label = milestoneLabel(event);
    if (!label) continue;
    const p = event.payload ?? {};
    const target = event.event_type === "approval.recorded" ? (publicId(p.target_id) ? p.target_id : p.target) : p.task_id;
    stops.push({ seq: event.event_seq, label, target: publicId(target) ? target : undefined });
  }
  return (cursor: number) => {
    // First stop at or after the cursor, including when stepping from a routine event.
    let lo = 0;
    let hi = stops.length;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (stops[mid].seq < cursor) lo = mid + 1;
      else hi = mid;
    }
    const current = stops[lo]?.seq === cursor ? stops[lo] : undefined;
    return { previous: stops[lo - 1]?.seq, current, next: stops[lo + (current ? 1 : 0)]?.seq };
  };
}
