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

  const chronological = [...iterations].sort((a, b) => b.iteration_index - a.iteration_index);

  return (
    <ol className="space-y-4">
      {chronological.map((it) => (
        <li key={it.iteration_index} className="border-l-2 border-[var(--accent)]/40 pl-4">
          <div className="mb-1 flex items-baseline gap-2">
            <span className="text-sm font-semibold text-white">Iteration {it.iteration_index}</span>
            <span className="text-xs text-[var(--text-muted)]">{fmtSeconds(it.elapsed_s)}</span>
          </div>
          {it.reasoning && (
            <p className="text-sm leading-relaxed text-[var(--text-secondary)]">{it.reasoning}</p>
          )}
          {it.changes_summary && (
            <p className="mt-1 text-xs italic text-[var(--text-muted)]">{it.changes_summary}</p>
          )}
        </li>
      ))}
    </ol>
  );
}
