import type { Genome } from "@/lib/types";

export interface PlanStats {
  chunks: number;
  durationSeconds: number;
  positiveStyles: number;
  negativeStyles: number;
  hasLyrics: boolean;
}

export function planStats(plan: Genome): PlanStats {
  const durationMs = plan.chunks.reduce((sum, chunk) => sum + chunk.duration_ms, 0);
  return {
    chunks: plan.chunks.length,
    durationSeconds: Math.round(durationMs / 1000),
    positiveStyles: plan.chunks.reduce((sum, chunk) => sum + chunk.positive_styles.length, 0),
    negativeStyles: plan.chunks.reduce((sum, chunk) => sum + chunk.negative_styles.length, 0),
    hasLyrics: plan.chunks.some((chunk) => {
      const text = chunk.text.trim();
      return text.length > 0 && !/^\[[^\]]+\]$/.test(text);
    }),
  };
}

/** Short human-readable summary of a plan: chunk count, total duration, and
 * the first chunk's section label — there are no global musical knobs (bpm,
 * key, etc.) on the plan anymore, only chunks. */
export function planSummary(plan: Genome): string {
  const stats = planStats(plan);
  const firstLabel = plan.chunks[0]?.text ?? "";
  return `${stats.chunks} chunk${stats.chunks === 1 ? "" : "s"} · ${stats.durationSeconds}s${
    firstLabel ? ` · starts "${firstLabel}"` : ""
  }`;
}
