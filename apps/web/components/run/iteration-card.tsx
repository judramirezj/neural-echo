import type { IterationResult } from "@/lib/types";
import { artifactUrl } from "@/lib/api";
import { fmtSeconds } from "@/lib/format";
import { ScoreMeter } from "./score-meter";
import { WorstRegions } from "./worst-regions";
import { planStats, planSummary } from "./plan-summary";
import { Badge } from "./badge";

interface IterationCardProps {
  jobId: string;
  iteration: IterationResult;
  domainMax: number;
  bestSoFar: number | null;
}

export function IterationCard({ jobId, iteration, domainMax, bestSoFar }: IterationCardProps) {
  const stats = planStats(iteration.plan);
  return (
    <div
      className={`rounded-md border p-3 ${
        iteration.is_best
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
          {iteration.is_best && <Badge tone="good">best</Badge>}
        </div>
      </div>

      {iteration.changes_summary && (
        <p className="mb-2 text-xs leading-relaxed text-[var(--text-secondary)]">{iteration.changes_summary}</p>
      )}

      {iteration.audio_path && (
        <audio
          controls
          preload="none"
          src={artifactUrl(jobId, iteration.audio_path)}
          className="mb-3 h-8 w-full"
        />
      )}
      {iteration.cost && <ScoreMeter value={iteration.cost.global_score} domainMax={domainMax} bestSoFar={bestSoFar} />}

      <div className="mt-3 flex flex-wrap gap-1.5">
        <Badge tone="muted">{fmtSeconds(iteration.elapsed_s)}</Badge>
        <Badge tone="muted">{stats.chunks} chunks</Badge>
        <Badge tone="muted">{stats.positiveStyles + stats.negativeStyles} style cues</Badge>
        {stats.hasLyrics && <Badge tone="muted">vocals</Badge>}
      </div>

      {iteration.cost && (
        <div className="mt-3 border-t border-white/5 pt-3">
          <WorstRegions cost={iteration.cost} />
        </div>
      )}
    </div>
  );
}
