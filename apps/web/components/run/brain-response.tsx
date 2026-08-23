"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { getBrainVisualization } from "@/lib/api";
import type { BrainVisualizationResponse } from "@/lib/types";

const BrainPlot = dynamic(() => import("./brain-plot"), {
  ssr: false,
  loading: () => <BrainLoading label="Initializing cortical surface…" />,
});

interface BrainResponseProps {
  jobId: string;
  iterationCount: number;
  isLive?: boolean;
}

export function BrainResponse({ jobId, iterationCount, isLive = false }: BrainResponseProps) {
  const [visualization, setVisualization] = useState<BrainVisualizationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (iterationCount === 0) return;
    const controller = new AbortController();
    getBrainVisualization(jobId, controller.signal)
      .then((data) => {
        setVisualization(data);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Brain response unavailable");
      });
    return () => controller.abort();
  }, [jobId, iterationCount]);

  const progress = useMemo(() => {
    const frames = visualization?.meta.frames ?? [];
    if (frames.length < 2) return null;
    const first = frames[0].mean_mismatch;
    const latest = frames.at(-1)?.mean_mismatch ?? first;
    if (first <= 0) return null;
    return ((first - latest) / first) * 100;
  }, [visualization]);

  const latest = visualization?.meta.frames.at(-1) ?? null;

  return (
    <section className="brain-stage relative isolate overflow-hidden rounded-2xl border border-white/[0.12] bg-[#090b12] shadow-[0_28px_90px_rgba(0,0,0,0.38)]">
      <div className="pointer-events-none absolute inset-0 opacity-70" aria-hidden="true">
        <div className="absolute -left-24 top-1/3 h-72 w-72 rounded-full bg-cyan-400/[0.07] blur-3xl" />
        <div className="absolute -right-24 top-8 h-72 w-72 rounded-full bg-rose-500/[0.07] blur-3xl" />
        <div className="brain-grid absolute inset-0" />
      </div>

      <header className="relative z-10 flex flex-col gap-5 border-b border-white/[0.08] px-5 py-5 sm:flex-row sm:items-start sm:justify-between sm:px-7">
        <div className="max-w-2xl">
          <div className="mb-2 flex items-center gap-2.5">
            <span className="relative flex h-2 w-2">
              {isLive && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />}
              <span className={`relative inline-flex h-2 w-2 rounded-full ${isLive ? "bg-emerald-400" : "bg-[var(--accent)]"}`} />
            </span>
            <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-[#8d93a6]">
              {isLive ? "Live neural convergence" : "Neural convergence replay"}
            </p>
          </div>
          <h2 className="text-xl font-semibold tracking-[-0.02em] text-white sm:text-2xl">
            Where the candidate still differs
          </h2>
          <p className="mt-2 max-w-xl text-xs leading-relaxed text-[#9298a9] sm:text-sm">
            Each glow is a cortical mismatch against the reference. As the optimizer learns,
            the residual signal should cool, soften, and disappear.
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-5 rounded-xl border border-white/[0.08] bg-white/[0.035] px-4 py-3 backdrop-blur">
          <Legend color="bg-[#41cfff]" glow="shadow-[0_0_12px_#41cfff]" label="Candidate lower" />
          <div className="h-7 w-px bg-white/10" />
          <Legend color="bg-[#ff5b68]" glow="shadow-[0_0_12px_#ff5b68]" label="Candidate higher" />
        </div>
      </header>

      <div className="relative min-h-[460px] sm:min-h-[560px]">
        {visualization ? (
          <BrainPlot figure={visualization.figure} />
        ) : iterationCount === 0 ? (
          <BrainLoading label="The brain map appears after the first candidate is scored…" />
        ) : error ? (
          <div className="absolute inset-0 grid place-items-center px-6 text-center">
            <div>
              <p className="text-sm text-[#aeb4c5]">{error}</p>
              <p className="mt-2 text-xs text-[#686f82]">The optimization continues while this view catches up.</p>
            </div>
          </div>
        ) : (
          <BrainLoading label="Mapping residuals onto 20,484 cortical vertices…" />
        )}

      </div>

      <footer className="relative z-10 grid border-t border-white/[0.08] bg-white/[0.018] sm:grid-cols-[1fr_auto]">
        <div className="grid grid-cols-3 divide-x divide-white/[0.08]">
          <Stat label="Frames mapped" value={visualization ? String(visualization.meta.frames.length).padStart(2, "0") : "—"} />
          <Stat label="Active surface" value={latest ? `${(latest.active_fraction * 100).toFixed(0)}%` : "—"} />
          <Stat
            label="Residual change"
            value={progress === null ? "—" : `${progress >= 0 ? "↓" : "↑"} ${Math.abs(progress).toFixed(0)}%`}
            tone={progress !== null && progress >= 0 ? "good" : "neutral"}
          />
        </div>
        <div className="flex items-center justify-center border-t border-white/[0.08] px-6 py-4 text-[10px] uppercase tracking-[0.18em] text-[#686f82] sm:border-l sm:border-t-0">
          Drag to orbit · Scroll to zoom
        </div>
      </footer>
    </section>
  );
}

function Legend({ color, glow, label }: { color: string; glow: string; label: string }) {
  return (
    <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-[#9ba2b5]">
      <span className={`h-2 w-2 rounded-full ${color} ${glow}`} />
      {label}
    </div>
  );
}

function Stat({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "good" | "neutral" }) {
  return (
    <div className="px-4 py-4 sm:px-6">
      <p className="text-[9px] uppercase tracking-[0.16em] text-[#686f82]">{label}</p>
      <p className={`mt-1 font-mono text-sm font-medium ${tone === "good" ? "text-emerald-400" : "text-[#e7eaf2]"}`}>
        {value}
      </p>
    </div>
  );
}

function BrainLoading({ label }: { label: string }) {
  return (
    <div className="absolute inset-0 grid place-items-center">
      <div className="text-center">
        <div className="brain-loader mx-auto mb-5 h-24 w-32 rounded-[50%] border border-white/[0.08]" />
        <p className="text-xs text-[#7c8396]">{label}</p>
      </div>
    </div>
  );
}
