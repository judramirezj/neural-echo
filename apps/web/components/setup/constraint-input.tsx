"use client";

const SUGGESTIONS = ["forest sounds", "no vocals", "lo-fi", "orchestral"];

interface ConstraintInputProps {
  value: string;
  onChange: (value: string) => void;
}

export function ConstraintInput({ value, onChange }: ConstraintInputProps) {
  return (
    <div>
      <textarea
        rows={3}
        placeholder='e.g. "use natural forest sounds, no percussion"'
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full resize-none rounded-xl border border-white/10 bg-black/20 px-4 py-3 text-sm text-white placeholder:text-[var(--text-muted)] outline-none transition focus:border-violet-300/50 focus:ring-2 focus:ring-violet-300/10"
      />
      <div className="mt-2 flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onChange(value.trim() ? `${value.trim()}, ${s}` : s)}
            className="rounded-full border border-white/10 bg-white/[0.035] px-3 py-1 text-xs text-[var(--text-secondary)] transition hover:border-violet-300/30 hover:bg-violet-300/[0.06] hover:text-white"
          >
            + {s}
          </button>
        ))}
      </div>
    </div>
  );
}
