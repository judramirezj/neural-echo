"use client";

import { useState } from "react";

interface AdvancedSettingsProps {
  maxGenerations: number;
  onMaxGenerationsChange: (v: number) => void;
  batchSize: number;
  onBatchSizeChange: (v: number) => void;
  adherenceTau: number;
  onAdherenceTauChange: (v: number) => void;
}

export function AdvancedSettings({
  maxGenerations,
  onMaxGenerationsChange,
  batchSize,
  onBatchSizeChange,
  adherenceTau,
  onAdherenceTauChange,
}: AdvancedSettingsProps) {
  const [open, setOpen] = useState(false);
  const totalRenders = maxGenerations * batchSize;

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
            label="Generations"
            hint="How many optimizer rounds to run before stopping"
            value={maxGenerations}
            min={1}
            max={20}
            onChange={onMaxGenerationsChange}
          />
          <Field
            label="Batch size"
            hint="How many candidate genomes the LLM proposes per generation"
            value={batchSize}
            min={1}
            max={30}
            onChange={onBatchSizeChange}
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
            Budget: up to {totalRenders} renders total ({maxGenerations} generations ×{" "}
            {batchSize} candidates).
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
