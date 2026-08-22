import type { Candidate } from "@/lib/types";
import { artifactUrl } from "@/lib/api";
import { fmtNum, fmtPercent } from "@/lib/format";
import { DBrainMeter } from "./d-brain-meter";
import { RadarChart } from "./radar-chart";

interface BestPanelProps {
  jobId: string;
  best: Candidate;
  bestGenerationIndex: number;
  domainMax: number;
  nullMedian: number | null;
  noiseFloor: number | null;
}

export function BestPanel({
  jobId,
  best,
  bestGenerationIndex,
  domainMax,
  nullMedian,
  noiseFloor,
}: BestPanelProps) {
  return (
    <div className="rounded-lg border border-[var(--accent)]/50 bg-[var(--surface-1)] p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-[var(--accent)]">
            Best so far
          </p>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            From generation {bestGenerationIndex}
            {best.percentile !== null && (
              <> · closer to reference than {fmtPercent(best.percentile)} of random pairs</>
            )}
          </p>
        </div>
        <p className="tabular-nums text-3xl font-semibold text-white">{fmtNum(best.D_brain)}</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div>
          {best.audio_path && (
            <audio
              controls
              src={artifactUrl(jobId, best.audio_path)}
              className="mb-4 w-full"
            />
          )}
          <DBrainMeter
            value={best.D_brain}
            domainMax={domainMax}
            nullMedian={nullMedian}
            noiseFloor={noiseFloor}
          />
          <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <dt className="text-[var(--text-muted)]">BPM / key</dt>
            <dd className="text-right text-[var(--text-secondary)]">
              {best.genome.bpm} · {best.genome.key_mode}
            </dd>
            <dt className="text-[var(--text-muted)]">Dynamic arc</dt>
            <dd className="text-right text-[var(--text-secondary)]">
              {best.genome.dynamic_arc.replace(/_/g, " ")}
            </dd>
            <dt className="text-[var(--text-muted)]">Adherence</dt>
            <dd className="text-right text-[var(--text-secondary)]">{fmtNum(best.adherence, 2)}</dd>
            <dt className="text-[var(--text-muted)]">Novelty sim</dt>
            <dd className="text-right text-[var(--text-secondary)]">
              {fmtNum(best.novelty_audio_sim, 2)}
            </dd>
          </dl>
        </div>
        <div>
          <p className="mb-2 text-xs font-medium text-[var(--text-secondary)]">
            Engagement profile vs. reference (7 networks)
          </p>
          <RadarChart deltas={best.per_network_deltas} />
        </div>
      </div>
    </div>
  );
}
