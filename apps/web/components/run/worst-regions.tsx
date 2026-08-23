import type { CostResult } from "@/lib/types";

interface WorstRegionsProps {
  cost: CostResult;
  topN?: number;
}

/**
 * A ranked bar list of the worst-scoring anatomical regions (score = distance
 * + (1 - arc correlation) between candidate and reference, lower is better),
 * plus the single worst cell and any notable left/right asymmetry. Replaces
 * the old 7-network radar chart — with ~50 regions a radar isn't legible, a
 * ranked list is.
 */
export function WorstRegions({ cost, topN = 6 }: WorstRegionsProps) {
  const worst = [...cost.regions].sort((a, b) => b.score - a.score).slice(0, topN);
  const maxScore = Math.max(0.1, ...worst.map((r) => r.score));
  const notableLaterality = Object.entries(cost.laterality)
    .filter(([, v]) => Math.abs(v) > 0.05)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3);

  return (
    <div>
      <p className="mb-2 text-xs font-medium text-[var(--text-secondary)]">
        Worst-scoring regions vs. reference
      </p>
      <ul className="space-y-1.5">
        {worst.map((r) => (
          <li key={r.region} className="flex items-center gap-2 text-xs">
            <span className="w-32 shrink-0 truncate text-[var(--text-muted)]" title={r.region}>
              {r.region}
            </span>
            <span className="relative h-2 flex-1 rounded-full bg-[var(--surface-2)]">
              <span
                className="absolute inset-y-0 left-0 rounded-full bg-[var(--diverging-neg)]"
                style={{ width: `${Math.min(100, (r.score / maxScore) * 100)}%` }}
              />
            </span>
            <span className="w-12 shrink-0 text-right tabular-nums text-[var(--text-secondary)]">
              {r.score.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>

      <p className="mt-3 text-[11px] text-[var(--text-muted)]">
        Worst single cell: window {cost.worst_cell.window_index + 1}, {cost.worst_cell.region} (
        {cost.worst_cell.difference > 0 ? "+" : ""}
        {cost.worst_cell.difference.toFixed(3)})
      </p>

      {notableLaterality.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {notableLaterality.map(([group, v]) => (
            <span
              key={group}
              title="Positive = left hemisphere scored worse than right"
              className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-[var(--text-muted)]"
            >
              {group} {v > 0 ? "L" : "R"}-skewed ({v > 0 ? "+" : ""}
              {v.toFixed(2)})
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
