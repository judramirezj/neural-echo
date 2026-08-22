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

/** Lowest D_brain among candidates seen so far (D_brain=null candidates are unscored/rejected). */
export function bestByDBrain<T extends { D_brain: number | null }>(
  candidates: T[]
): T | null {
  let best: T | null = null;
  for (const c of candidates) {
    if (c.D_brain === null) continue;
    if (best === null || best.D_brain === null || c.D_brain < best.D_brain) {
      best = c;
    }
  }
  return best;
}
