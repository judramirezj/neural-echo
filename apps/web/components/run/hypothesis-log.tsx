import type { GenerationCompleteEvent } from "@/lib/types";
import { fmtSeconds } from "@/lib/format";

interface HypothesisLogProps {
  generations: GenerationCompleteEvent[];
}

export function HypothesisLog({ generations }: HypothesisLogProps) {
  if (generations.length === 0) {
    return (
      <p className="text-sm text-[var(--text-muted)]">
        Waiting for the first generation to complete…
      </p>
    );
  }

  const chronological = [...generations].sort((a, b) => b.generation_index - a.generation_index);
  const latest = chronological[0];

  return (
    <div>
      {latest.learned_insights && (
        <p className="mb-4 rounded-md border border-white/10 bg-[var(--surface-2)] px-3 py-2 text-xs text-[var(--text-secondary)]">
          <span className="font-medium text-[var(--text-primary)]">Notes carried forward — </span>
          {latest.learned_insights}
        </p>
      )}
      <ol className="space-y-4">
        {chronological.map((g) => (
          <li key={g.generation_index} className="border-l-2 border-[var(--accent)]/40 pl-4">
            <div className="mb-1 flex items-baseline gap-2">
              <span className="text-sm font-semibold text-white">Generation {g.generation_index}</span>
              <span className="text-xs text-[var(--text-muted)]">{fmtSeconds(g.elapsed_s)}</span>
            </div>
            <p className="text-sm leading-relaxed text-[var(--text-secondary)]">{g.hypothesis}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
