import type { ReactNode } from "react";

type Tone = "good" | "critical" | "muted";

const TONE_CLASSES: Record<Tone, string> = {
  good: "border-[var(--status-good)]/40 bg-[var(--status-good)]/10 text-[var(--status-good)]",
  critical:
    "border-[var(--status-critical)]/40 bg-[var(--status-critical)]/10 text-[var(--status-critical)]",
  muted: "border-white/10 bg-white/5 text-[var(--text-muted)]",
};

export function Badge({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}
