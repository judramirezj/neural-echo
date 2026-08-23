import type { IterationResult } from "@/lib/types";
import { fmtSeconds } from "@/lib/format";

interface HypothesisLogProps {
  iterations: IterationResult[];
}

export function HypothesisLog({ iterations }: HypothesisLogProps) {
  if (iterations.length === 0) {
    return (
      <p className="text-sm text-[var(--text-muted)]">
        Waiting for the first iteration to complete…
      </p>
    );
  }

  const newestFirst = [...iterations].sort((a, b) => b.iteration_index - a.iteration_index);

  return (
    <ol className="space-y-4">
      {newestFirst.map((it) => (
        <li key={it.iteration_index} className="rounded-xl border border-white/[0.08] bg-black/15 p-4">
          <div className="mb-1 flex items-baseline gap-2">
            <span className="grid h-7 w-7 place-items-center rounded-full bg-violet-300/10 font-mono text-[10px] font-semibold text-violet-200">{it.iteration_index}</span>
            <span className="text-sm font-semibold text-white">Creative decision</span>
            <span className="text-xs text-[var(--text-muted)]">{fmtSeconds(it.elapsed_s)}</span>
          </div>
          {it.changes_summary && (
            <p className="mt-3 text-sm font-medium leading-relaxed text-white/90">{it.changes_summary}</p>
          )}
          {it.reasoning && (
            <div className="mt-3 border-t border-white/[0.06] pt-3">
              <p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">Why it chose this</p>
              <p className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">{it.reasoning}</p>
            </div>
          )}
        </li>
      ))}
    </ol>
  );
}
