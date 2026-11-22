import type { RecordedAttemptArtifacts, TaskArtifacts } from "../api/types";
import { Card, DiffBlock, Notice } from "./ui";

export function RecordedArtifactCards({
  artifacts,
  manifest,
  showEmpty = true,
}: {
  artifacts: Partial<TaskArtifacts | RecordedAttemptArtifacts>;
  manifest: string | null | undefined;
  showEmpty?: boolean;
}) {
  const verdicts = (artifacts.verdicts || []).map((verdict) => JSON.stringify(verdict, null, 2)).join("\n\n");
  const logs = Object.entries(artifacts.check_logs || {});
  const cards = [
    showEmpty || artifacts.diff_patch ? (
      <Card title="diff.patch" note="隔离沙盒产物" key="diff">
        <DiffBlock patch={artifacts.diff_patch} />
      </Card>
    ) : null,
    showEmpty || manifest ? (
      <Card title="MANIFEST.md" note="完整性清单" key="manifest">
        <pre className="code-block scroller">{manifest || "（暂无 MANIFEST.md）"}</pre>
      </Card>
    ) : null,
    showEmpty || verdicts ? (
      <Card title="Argus 独立复核" note="冷启动审查" key="verdicts">
        <pre className="code-block scroller">{verdicts || "（暂无复核记录）"}</pre>
      </Card>
    ) : null,
    ...logs.map(([id, log]) => (
      <Card title={`检查日志 · ${id}`} key={`log-${id}`}>
        <pre className="code-block scroller">{log}</pre>
      </Card>
    )),
  ].filter(Boolean);
  return cards.length ? <>{cards}</> : <Notice level="info">此版本的执行文件不可用</Notice>;
}
