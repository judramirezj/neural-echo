import type { IterationResult } from "@/lib/types";
import { artifactUrl } from "@/lib/api";
import { fmtNum } from "@/lib/format";
import { ScoreMeter } from "./score-meter";
import { WorstRegions } from "./worst-regions";
import { planStats, planSummary } from "./plan-summary";

interface BestPanelProps {
  jobId: string;
  best: IterationResult;
  domainMax: number;
}

export function BestPanel({ jobId, best, domainMax }: BestPanelProps) {
  const stats = planStats(best.plan);
  return (
    <div className="rounded-lg border border-[var(--accent)]/50 bg-[var(--surface-1)] p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-[var(--accent)]">
            Best so far
          </p>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            From iteration {best.iteration_index}
          </p>
        </div>
        <p className="tabular-nums text-3xl font-semibold text-white">
          {best.cost ? fmtNum(best.cost.global_score) : "—"}
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div>
          {best.audio_path && (
            <audio controls src={artifactUrl(jobId, best.audio_path)} className="mb-4 w-full" />
          )}
          {best.audio_path && (
            <a
              href={artifactUrl(jobId, best.audio_path)}
              download
              className="mb-4 inline-flex rounded-lg border border-white/15 bg-white/[0.04] px-3 py-2 text-xs font-semibold text-white transition hover:border-violet-300/35"
            >
              Download this version ↓
            </a>
          )}
          {best.cost && <ScoreMeter value={best.cost.global_score} domainMax={domainMax} bestSoFar={best.cost.global_score} />}
          <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <dt className="text-[var(--text-muted)]">Plan</dt>
            <dd className="text-right text-[var(--text-secondary)]">{planSummary(best.plan)}</dd>
            <dt className="text-[var(--text-muted)]">Prompt complexity</dt>
            <dd className="text-right text-[var(--text-secondary)]">
              {stats.positiveStyles + stats.negativeStyles} style cues
            </dd>
          </dl>
        </div>
        <div>{best.cost && <WorstRegions cost={best.cost} />}</div>
      </div>
    </div>
  );
}
