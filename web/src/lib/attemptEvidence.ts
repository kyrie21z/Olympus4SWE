import type { RecordedAttemptArtifacts } from "../api/types";
import { publicId } from "./evidenceId";

type CheckOutcome = "通过" | "未通过" | "结果未记录";

export function summarizeAttempt(art: RecordedAttemptArtifacts, taskId: string, attempt: number) {
  const checks = new Map<string, CheckOutcome>();
  const sections = art.manifest_md.split(/^## Acceptance checks[^\S\n]*$/m);
  const manifest = sections.length > 1 ? sections[sections.length - 1].split(/^## /m)[0] : "";
  for (const line of manifest.split(/\r?\n/)) {
    const row = /^- (\S+) \((?:command|independent_review)\): (passed|not passed)(?: —|$)/.exec(line);
    if (row && publicId(row[1])) checks.set(row[1], row[2] === "passed" ? "通过" : "未通过");
  }
  for (const [name, log] of Object.entries(art.check_logs)) {
    if (!publicId(name)) continue;
    // The command runner writes this header before either output stream.
    const header = /^command: [^\r\n]*\r?\nexit_code: (-?\d+)\r?\n--- stdout ---/.exec(log);
    const unnumbered = name.replace(/^\d+-/, "");
    const id = checks.has(unnumbered) ? unnumbered : name;
    if (header) checks.set(id, Number(header[1]) === 0 ? "通过" : "未通过");
    else if (!checks.has(id)) checks.set(id, "结果未记录");
  }
  const files = new Set<string>();
  let added = 0;
  let removed = 0;
  let inHunk = false;
  const lines = art.diff_patch.split(/\r?\n/);
  for (const [i, line] of lines.entries()) {
    if (line.startsWith("diff --git ")) inHunk = false;
    if (line.startsWith("--- ") && lines[i + 1]?.startsWith("+++ ")) {
      inHunk = false;
      const target = lines[i + 1].slice(4).split("\t")[0];
      const source = target === "/dev/null" ? line.slice(4).split("\t")[0] : target;
      const path = source.replace(/^[ab]\//, "");
      if (path.length <= 160 && path.split("/").every(part => part !== "." && part !== ".." && publicId(part))) {
        files.add(path);
      }
    } else if (line.startsWith("@@ ")) inHunk = true;
    else if (inHunk && line.startsWith("+")) added++;
    else if (inHunk && line.startsWith("-")) removed++;
  }
  const reviews = art.verdicts.filter(v => v.task_id === taskId && v.attempt === attempt);
  return {
    checks: [...checks], files: [...files], added, removed, hasPatch: Boolean(art.diff_patch.trim()),
    reviewCount: reviews.length, reviewsPassed: reviews.filter(v => v.passed).length,
  };
}
