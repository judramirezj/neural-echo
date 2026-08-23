import type { IterationResult } from "@/lib/types";
import { artifactUrl } from "@/lib/api";
import { fmtNum, fmtSeconds } from "@/lib/format";
import { ScoreMeter } from "./score-meter";
import { WorstRegions } from "./worst-regions";
import { planSummary } from "./plan-summary";
import { Badge } from "./badge";

const REJECTED_LABEL: Record<string, string> = {
  generation_failed: "Render failed",
  near_cover: "Rejected — near-cover of reference",
  constraint_not_met: "Rejected — constraint not met",
};

interface IterationCardProps {
  jobId: string;
  iteration: IterationResult;
  domainMax: number;
  bestSoFar: number | null;
}

export function IterationCard({ jobId, iteration, domainMax, bestSoFar }: IterationCardProps) {
  const rejected = iteration.rejected_reason !== null;

  return (
    <div
      className={`rounded-md border p-3 ${
        rejected
          ? "border-white/5 bg-[var(--surface-1)]/40 opacity-60"
          : iteration.is_best
          ? "border-[var(--accent)]/60 bg-[var(--surface-1)]"
          : "border-white/10 bg-[var(--surface-1)]"
      }`}
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-white">
            Iteration {iteration.iteration_index}
          </p>
          <p className="truncate text-xs text-[var(--text-muted)]" title={planSummary(iteration.plan)}>
            {planSummary(iteration.plan)}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {iteration.is_best && !rejected && <Badge tone="good">best</Badge>}
          {rejected && <Badge tone="critical">rejected</Badge>}
        </div>
      </div>

      {iteration.changes_summary && (
        <p className="mb-2 text-xs leading-relaxed text-[var(--text-secondary)]">{iteration.changes_summary}</p>
      )}

      {rejected ? (
        <p className="rounded bg-black/20 px-2 py-2 text-xs text-[var(--status-critical)]">
          {REJECTED_LABEL[iteration.rejected_reason ?? ""] ?? iteration.rejected_reason}
        </p>
      ) : (
        <>
          {iteration.audio_path && (
            <audio
              controls
              preload="none"
              src={artifactUrl(jobId, iteration.audio_path)}
              className="mb-3 h-8 w-full"
            />
          )}
          {iteration.cost && <ScoreMeter value={iteration.cost.global_score} domainMax={domainMax} bestSoFar={bestSoFar} />}
        </>
      )}

      <div className="mt-3 flex flex-wrap gap-1.5">
        {iteration.adherence !== null && (
          <Badge tone={iteration.rejected_reason !== "constraint_not_met" ? "good" : "critical"}>
            adherence {fmtNum(iteration.adherence, 2)}
          </Badge>
        )}
        {iteration.novelty_audio_sim !== null && (
          <Badge tone={iteration.is_near_cover ? "critical" : "muted"}>
            novelty sim {fmtNum(iteration.novelty_audio_sim, 2)}
            {iteration.is_near_cover ? " · near-cover" : ""}
          </Badge>
        )}
        <Badge tone="muted">{fmtSeconds(iteration.elapsed_s)}</Badge>
      </div>

      {iteration.cost && !rejected && (
        <div className="mt-3 border-t border-white/5 pt-3">
          <WorstRegions cost={iteration.cost} />
        </div>
      )}
    </div>
  );
}
