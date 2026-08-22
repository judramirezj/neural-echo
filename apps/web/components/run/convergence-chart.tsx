"use client";

import { useRef, useState } from "react";
import { fmtNum } from "@/lib/format";

export interface ConvergencePoint {
  generationIndex: number;
  bestSoFar: number | null;
  meanDBrain: number | null;
}

interface ConvergenceChartProps {
  points: ConvergencePoint[];
  nullMedian: number | null;
  noiseFloor: number | null;
}

const WIDTH = 640;
const HEIGHT = 260;
const MARGIN = { top: 16, right: 56, bottom: 28, left: 40 };
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;

export function ConvergenceChart({ points, nullMedian, noiseFloor }: ConvergenceChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const maxGen = Math.max(1, ...points.map((p) => p.generationIndex));
  const allValues = points
    .flatMap((p) => [p.bestSoFar, p.meanDBrain])
    .filter((v): v is number => v !== null);
  const domainCandidates = [...allValues];
  if (nullMedian !== null) domainCandidates.push(nullMedian);
  if (noiseFloor !== null) domainCandidates.push(noiseFloor);
  const maxVal = domainCandidates.length > 0 ? Math.max(...domainCandidates) * 1.15 : 1;
  const minVal = 0;

  const x = (gen: number) => (gen / maxGen) * PLOT_W;
  const y = (v: number) => PLOT_H - ((v - minVal) / (maxVal - minVal || 1)) * PLOT_H;

  const bestPath = buildPath(points, "bestSoFar", x, y);
  const meanPath = buildPath(points, "meanDBrain", x, y);

  const lastBest = [...points].reverse().find((p) => p.bestSoFar !== null);
  const lastMean = [...points].reverse().find((p) => p.meanDBrain !== null);

  function handleMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!svgRef.current || points.length === 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * WIDTH - MARGIN.left;
    const gen = Math.round((px / PLOT_W) * maxGen);
    const clamped = Math.min(maxGen, Math.max(0, gen));
    const idx = points.findIndex((p) => p.generationIndex === clamped);
    setHoverIdx(idx >= 0 ? idx : null);
  }

  const hoverPoint = hoverIdx !== null ? points[hoverIdx] : null;
  const yTicks = niceTicks(minVal, maxVal, 4);

  return (
    <div className="relative">
      <div className="mb-2 flex items-center gap-4 text-xs">
        <LegendKey color="var(--accent)" label="Best D_brain so far" />
        <LegendKey color="var(--series-2)" label="Mean D_brain (this generation)" />
        {noiseFloor !== null && <LegendKey color="var(--status-good)" dashed label="Noise floor" />}
        {nullMedian !== null && <LegendKey color="var(--text-muted)" dashed label="Random-pair median" />}
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full"
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIdx(null)}
        role="img"
        aria-label="Convergence of best and mean D_brain across generations"
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* gridlines */}
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={0} x2={PLOT_W} y1={y(t)} y2={y(t)} stroke="var(--gridline)" strokeWidth={1} />
              <text x={-8} y={y(t)} textAnchor="end" dominantBaseline="middle" fontSize={10} fill="var(--text-muted)">
                {t}
              </text>
            </g>
          ))}

          {/* x axis ticks */}
          {points.map((p) => (
            <text
              key={p.generationIndex}
              x={x(p.generationIndex)}
              y={PLOT_H + 18}
              textAnchor="middle"
              fontSize={10}
              fill="var(--text-muted)"
            >
              {p.generationIndex}
            </text>
          ))}

          {/* reference lines */}
          {noiseFloor !== null && (
            <line
              x1={0}
              x2={PLOT_W}
              y1={y(noiseFloor)}
              y2={y(noiseFloor)}
              stroke="var(--status-good)"
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
          )}
          {nullMedian !== null && (
            <line
              x1={0}
              x2={PLOT_W}
              y1={y(nullMedian)}
              y2={y(nullMedian)}
              stroke="var(--text-muted)"
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
          )}

          {/* series */}
          <path d={meanPath} fill="none" stroke="var(--series-2)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          <path d={bestPath} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />

          {points.map((p) =>
            p.bestSoFar !== null ? (
              <circle key={`b${p.generationIndex}`} cx={x(p.generationIndex)} cy={y(p.bestSoFar)} r={4} fill="var(--accent)" stroke="var(--surface-2)" strokeWidth={2} />
            ) : null
          )}
          {points.map((p) =>
            p.meanDBrain !== null ? (
              <circle key={`m${p.generationIndex}`} cx={x(p.generationIndex)} cy={y(p.meanDBrain)} r={4} fill="var(--series-2)" stroke="var(--surface-2)" strokeWidth={2} />
            ) : null
          )}

          {/* end labels */}
          {lastBest?.bestSoFar != null && (
            <text x={x(lastBest.generationIndex) + 8} y={y(lastBest.bestSoFar)} fontSize={11} fill="var(--text-primary)" dominantBaseline="middle">
              {fmtNum(lastBest.bestSoFar)}
            </text>
          )}
          {lastMean?.meanDBrain != null && (
            <text x={x(lastMean.generationIndex) + 8} y={y(lastMean.meanDBrain)} fontSize={11} fill="var(--text-secondary)" dominantBaseline="middle">
              {fmtNum(lastMean.meanDBrain)}
            </text>
          )}

          {/* crosshair */}
          {hoverPoint && (
            <line x1={x(hoverPoint.generationIndex)} x2={x(hoverPoint.generationIndex)} y1={0} y2={PLOT_H} stroke="var(--text-muted)" strokeWidth={1} />
          )}
        </g>
      </svg>

      {hoverPoint && (
        <div
          className="pointer-events-none absolute rounded border border-white/10 bg-[var(--surface-2)] px-2.5 py-1.5 text-xs shadow-lg"
          style={{
            left: `${((MARGIN.left + x(hoverPoint.generationIndex)) / WIDTH) * 100}%`,
            top: 0,
            transform: "translate(-50%, -100%)",
          }}
        >
          <p className="mb-1 font-medium text-white">Generation {hoverPoint.generationIndex}</p>
          <p className="text-[var(--accent)]">best: {fmtNum(hoverPoint.bestSoFar)}</p>
          <p className="text-[var(--series-2)]">mean: {fmtNum(hoverPoint.meanDBrain)}</p>
        </div>
      )}
    </div>
  );
}

function buildPath(
  points: ConvergencePoint[],
  key: "bestSoFar" | "meanDBrain",
  x: (g: number) => number,
  y: (v: number) => number
): string {
  const segs: string[] = [];
  let started = false;
  for (const p of points) {
    const v = p[key];
    if (v === null) continue;
    segs.push(`${started ? "L" : "M"}${x(p.generationIndex)},${y(v)}`);
    started = true;
  }
  return segs.join(" ");
}

function niceTicks(min: number, max: number, count: number): number[] {
  if (max <= min) return [min];
  const step = (max - min) / count;
  const ticks: number[] = [];
  for (let i = 0; i <= count; i++) {
    ticks.push(Number((min + step * i).toFixed(2)));
  }
  return ticks;
}

function LegendKey({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-[var(--text-secondary)]">
      <svg width="16" height="8" aria-hidden="true">
        <line
          x1="0"
          y1="4"
          x2="16"
          y2="4"
          stroke={color}
          strokeWidth="2"
          strokeDasharray={dashed ? "3 2" : undefined}
        />
      </svg>
      {label}
    </span>
  );
}
