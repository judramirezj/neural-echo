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
        className="w-full resize-none rounded-md border border-white/10 bg-[var(--surface-1)] px-3 py-2.5 text-sm text-white placeholder:text-[var(--text-muted)] outline-none focus:border-[var(--accent)]"
      />
      <div className="mt-2 flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onChange(value.trim() ? `${value.trim()}, ${s}` : s)}
            className="rounded-full border border-white/10 bg-[var(--surface-1)] px-3 py-1 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--accent)] hover:text-white"
          >
            + {s}
          </button>
        ))}
      </div>
    </div>
  );
}
