import type { IterationResult } from "@/lib/types";
import { fmtNum } from "@/lib/format";

export function OptimizationInsights({ iterations }: { iterations: IterationResult[] }) {
  const scored = [...iterations]
    .filter((iteration) => iteration.cost !== null)
    .sort((a, b) => a.iteration_index - b.iteration_index);
  if (scored.length === 0) return null;

  const first = scored[0];
  const latest = scored.at(-1) ?? first;
  const previous = scored.at(-2) ?? null;
  const best = scored.reduce((winner, iteration) =>
    iteration.cost!.global_score < winner.cost!.global_score ? iteration : winner
  );
  const improvement = first.cost!.global_score > 0
    ? ((first.cost!.global_score - best.cost!.global_score) / first.cost!.global_score) * 100
    : 0;
  const latestDelta = previous
    ? latest.cost!.global_score - previous.cost!.global_score
    : null;
  const worst = latest.cost!.worst_cell;
  const windowPosition = describeWindow(
    worst.window_index,
    latest.cost!.windows.length
  );

  return (
    <section className="overflow-hidden rounded-2xl border border-violet-300/15 bg-gradient-to-br from-violet-300/[0.08] via-[var(--surface-1)] to-cyan-300/[0.04]">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-white/[0.08] px-5 py-4 sm:px-6">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-violet-200/70">What the system is learning</p>
          <h2 className="mt-1 text-lg font-semibold text-white">The optimization, in plain English</h2>
        </div>
        <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-[10px] text-[var(--text-muted)]">
          Updated after iteration {latest.iteration_index}
        </span>
      </header>

      <div className="grid md:grid-cols-3">
        <Insight
          eyebrow="Overall pattern"
          title={improvement > 0.5 ? `${improvement.toFixed(0)}% closer so far` : "Establishing a baseline"}
          body={
            latestDelta === null
              ? "This first render gives the system a baseline for every brain region and moment in the track."
              : latestDelta < 0
                ? `The latest change improved the raw match by ${Math.abs(latestDelta).toFixed(3)}. Lower scores mean the predicted responses are moving closer.`
                : `The latest experiment moved ${latestDelta.toFixed(3)} away from the target. The best earlier plan is still preserved.`
          }
        />
        <Insight
          eyebrow="Current focus"
          title={`${humanizeRegion(worst.region)} · ${windowPosition}`}
          body={`This is the largest remaining mismatch. The candidate response is ${worst.difference >= 0 ? "stronger" : "weaker"} than the reference here, so it becomes high-priority feedback—not a claim of musical causation.`}
        />
        <Insight
          eyebrow="Latest creative decision"
          title={latest.changes_summary || "Testing the current musical hypothesis"}
          body={latest.reasoning || "The next decision is guided by the full region-by-time response matrix while keeping your creative direction intact."}
        />
      </div>

      <footer className="border-t border-white/[0.08] px-5 py-3 text-[10px] leading-relaxed text-[var(--text-muted)] sm:px-6">
        Selection criterion: raw spatial distance plus temporal-shape mismatch across {latest.cost!.regions.length} cortical regions and {latest.cost!.windows.length} moments. Best retained: iteration {best.iteration_index} at {fmtNum(best.cost!.global_score)}.
      </footer>
    </section>
  );
}

function Insight({ eyebrow, title, body }: { eyebrow: string; title: string; body: string }) {
  return (
    <article className="border-b border-white/[0.08] p-5 last:border-b-0 md:border-b-0 md:border-r md:last:border-r-0 sm:p-6">
      <p className="text-[9px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">{eyebrow}</p>
      <h3 className="mt-2 text-sm font-semibold leading-snug text-white">{title}</h3>
      <p className="mt-2 text-xs leading-relaxed text-[var(--text-secondary)]">{body}</p>
    </article>
  );
}

function humanizeRegion(region: string): string {
  return region
    .replace(/_(left|right)$/i, " ($1)")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function describeWindow(index: number, total: number): string {
  if (total <= 1) return "across the track";
  const position = index / (total - 1);
  if (position < 0.34) return "early in the track";
  if (position < 0.67) return "through the middle";
  return "near the ending";
}
