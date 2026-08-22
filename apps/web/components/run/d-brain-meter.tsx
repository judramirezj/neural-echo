import { fmtNum } from "@/lib/format";

interface DBrainMeterProps {
  value: number | null;
  domainMax: number;
  nullMedian: number | null;
  noiseFloor: number | null;
}

/**
 * A "ratio against a limit" meter. Lower D_brain is better, so the filled
 * track grows from the left (0 / perfect match) toward the value. Two
 * reference ticks overlay the same 0..domainMax scale: the null-pair median
 * ("random music") and the noise floor ("theoretical best" achievable
 * against this reference).
 */
export function DBrainMeter({ value, domainMax, nullMedian, noiseFloor }: DBrainMeterProps) {
  const safeMax = domainMax > 0 ? domainMax : 1;
  const pct = (v: number) => Math.min(100, Math.max(0, (v / safeMax) * 100));

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs text-[var(--text-muted)]">D_brain</span>
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
        {noiseFloor !== null && (
          <div
            title={`Noise floor: ${fmtNum(noiseFloor)}`}
            className="absolute top-[-3px] h-4 w-[2px] bg-[var(--status-good)]"
            style={{ left: `${pct(noiseFloor)}%` }}
          />
        )}
        {nullMedian !== null && (
          <div
            title={`Random-pair median: ${fmtNum(nullMedian)}`}
            className="absolute top-[-3px] h-4 w-[2px] border-l-2 border-dashed border-[var(--text-muted)]"
            style={{ left: `${pct(nullMedian)}%` }}
          />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-[var(--text-muted)]">
        <span>0 (perfect)</span>
        <span>
          {noiseFloor !== null && <>floor {fmtNum(noiseFloor, 2)} · </>}
          {nullMedian !== null && <>random {fmtNum(nullMedian, 2)}</>}
        </span>
      </div>
    </div>
  );
}
