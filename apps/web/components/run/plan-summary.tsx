import type { Genome } from "@/lib/types";

/** Short human-readable summary of a plan: chunk count, total duration, and
 * the first chunk's section label — there are no global musical knobs (bpm,
 * key, etc.) on the plan anymore, only chunks. */
export function planSummary(plan: Genome): string {
  const totalMs = plan.chunks.reduce((sum, c) => sum + c.duration_ms, 0);
  const totalS = Math.round(totalMs / 1000);
  const firstLabel = plan.chunks[0]?.text ?? "";
  return `${plan.chunks.length} chunk${plan.chunks.length === 1 ? "" : "s"} · ${totalS}s${
    firstLabel ? ` · starts "${firstLabel}"` : ""
  }`;
}
