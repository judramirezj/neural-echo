"use client";

import { useState } from "react";

interface AdvancedSettingsProps {
  maxIterations: number;
  onMaxIterationsChange: (v: number) => void;
  adherenceTau: number;
  onAdherenceTauChange: (v: number) => void;
}

export function AdvancedSettings({
  maxIterations,
  onMaxIterationsChange,
  adherenceTau,
  onAdherenceTauChange,
}: AdvancedSettingsProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-md border border-white/10 bg-[var(--surface-1)]">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 text-sm text-[var(--text-secondary)]"
      >
        <span>Advanced</span>
        <span className="text-[var(--text-muted)]">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="space-y-4 border-t border-white/10 px-4 py-4">
          <Field
            label="Iterations"
            hint="How many optimizer rounds to run before stopping — the optimizer refines one plan iteration by iteration"
            value={maxIterations}
            min={1}
            max={20}
            onChange={onMaxIterationsChange}
          />
          <Field
            label="Adherence threshold (τ)"
            hint="Minimum constraint-adherence score a candidate must clear to be scored"
            value={adherenceTau}
            min={0}
            max={1}
            step={0.01}
            onChange={onAdherenceTauChange}
          />
          <p className="text-xs text-[var(--text-muted)]">
            Budget: up to {maxIterations} renders total (one candidate per iteration).
          </p>
        </div>
      )}
    </div>
  );
}

function Field({
  label,
  hint,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <label className="text-xs font-medium text-white">{label}</label>
        <span className="tabular-nums text-xs text-[var(--text-secondary)]">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--accent)]"
      />
      <p className="mt-1 text-xs text-[var(--text-muted)]">{hint}</p>
    </div>
  );
}
