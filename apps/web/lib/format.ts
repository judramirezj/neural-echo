export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtPercent(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)}%`;
}

export function fmtSigned(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}`;
}

export function fmtSeconds(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)}s`;
}

/** Lowest global_score among iterations seen so far (cost=null iterations were rejected/unscored). */
export function bestByScore<T extends { cost: { global_score: number } | null }>(
  iterations: T[]
): T | null {
  let best: T | null = null;
  for (const it of iterations) {
    if (it.cost === null) continue;
    if (best === null || best.cost === null || it.cost.global_score < best.cost.global_score) {
      best = it;
    }
  }
  return best;
}
