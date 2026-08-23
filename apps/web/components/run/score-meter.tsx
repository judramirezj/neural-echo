import { fmtNum } from "@/lib/format";

interface ScoreMeterProps {
  value: number | null;
  domainMax: number;
  bestSoFar: number | null;
}

/**
 * A "ratio against a limit" meter. Lower global_score is better, so the
 * filled track grows from the left (0 / perfect match) toward the value.
 * domainMax and bestSoFar are derived from this run's own iterations (there's
 * no external calibration to compare against) — a tick marks the best score
 * seen so far in the run.
 */
export function ScoreMeter({ value, domainMax, bestSoFar }: ScoreMeterProps) {
  const safeMax = domainMax > 0 ? domainMax : 1;
  const pct = (v: number) => Math.min(100, Math.max(0, (v / safeMax) * 100));

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs text-[var(--text-muted)]">raw brain distance · lower is better</span>
        <span className="tabular-nums text-sm font-medium text-white">
          {value === null ? "—" : fmtNum(value)}
        </span>
      </div>
      <div className="relative h-2.5 w-full rounded-full bg-[var(--surface-2)]">
        {value !== null && (
          <div
            className="h-2.5 rounded-full bg-[var(--accent)]"
            style={{ width: `${pct(value)}%` }}
          />
        )}
        {bestSoFar !== null && (
          <div
            title={`Best so far: ${fmtNum(bestSoFar)}`}
            className="absolute top-[-3px] h-4 w-[2px] bg-[var(--status-good)]"
            style={{ left: `${pct(bestSoFar)}%` }}
          />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-[var(--text-muted)]">
        <span>0 (perfect)</span>
        {bestSoFar !== null && <span>best so far {fmtNum(bestSoFar, 2)}</span>}
      </div>
    </div>
  );
}
