"use client";

import { useId, useMemo, useState } from "react";

interface RadarChartProps {
  /** Sigma-unit engagement delta per brain network, candidate vs reference. */
  deltas: Record<string, number>;
}

const SIZE = 280;
const CENTER = SIZE / 2;
const R_MAX = 100; // outer radius (full scale)
const R_ZERO = R_MAX / 2; // radius that represents delta = 0 (the reference line)

export function RadarChart({ deltas }: RadarChartProps) {
  const gradId = useId();
  const [hovered, setHovered] = useState<string | null>(null);

  const entries = useMemo(() => Object.entries(deltas), [deltas]);
  const n = entries.length;

  const maxAbs = useMemo(() => {
    const m = Math.max(0.1, ...entries.map(([, v]) => Math.abs(v)));
    return m * 1.2;
  }, [entries]);

  if (n === 0) {
    return (
      <p className="text-xs text-[var(--text-muted)]">
        No per-network engagement data for this candidate yet.
      </p>
    );
  }

  const angleFor = (i: number) => (Math.PI * 2 * i) / n - Math.PI / 2;
  const radiusFor = (v: number) => R_ZERO + (v / maxAbs) * R_ZERO;

  const points = entries.map(([name, v], i) => {
    const angle = angleFor(i);
    const r = radiusFor(v);
    return {
      name,
      v,
      x: CENTER + r * Math.cos(angle),
      y: CENTER + r * Math.sin(angle),
      labelX: CENTER + (R_MAX + 22) * Math.cos(angle),
      labelY: CENTER + (R_MAX + 22) * Math.sin(angle),
    };
  });

  const polygonPoints = points.map((p) => `${p.x},${p.y}`).join(" ");
  const zeroCirclePoints = entries
    .map((_, i) => {
      const angle = angleFor(i);
      return `${CENTER + R_ZERO * Math.cos(angle)},${CENTER + R_ZERO * Math.sin(angle)}`;
    })
    .join(" ");

  const hoveredPoint = points.find((p) => p.name === hovered) ?? null;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full max-w-[280px]" role="img"
        aria-label="Per-network engagement delta of best candidate vs reference">
        <defs>
          <radialGradient id={gradId} cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.04" />
          </radialGradient>
        </defs>

        {/* spokes */}
        {entries.map((_, i) => {
          const angle = angleFor(i);
          return (
            <line
              key={i}
              x1={CENTER}
              y1={CENTER}
              x2={CENTER + R_MAX * Math.cos(angle)}
              y2={CENTER + R_MAX * Math.sin(angle)}
              stroke="var(--gridline)"
              strokeWidth={1}
            />
          );
        })}

        {/* zero (reference) line — the implicit reference profile */}
        <polygon
          points={zeroCirclePoints}
          fill="none"
          stroke="var(--baseline)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />

        {/* candidate polygon */}
        <polygon points={polygonPoints} fill={`url(#${gradId})`} stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" />

        {/* vertex markers */}
        {points.map((p) => (
          <g key={p.name}>
            <circle
              cx={p.x}
              cy={p.y}
              r={9}
              fill="transparent"
              onMouseEnter={() => setHovered(p.name)}
              onMouseLeave={() => setHovered((h) => (h === p.name ? null : h))}
              style={{ cursor: "pointer" }}
            />
            <circle
              cx={p.x}
              cy={p.y}
              r={4.5}
              fill={p.v >= 0 ? "var(--diverging-pos)" : "var(--diverging-neg)"}
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
          </g>
        ))}

        {/* labels */}
        {points.map((p) => (
          <text
            key={p.name}
            x={p.labelX}
            y={p.labelY}
            fontSize={9.5}
            fill="var(--text-secondary)"
            textAnchor={Math.cos(angleFor(points.indexOf(p))) > 0.2 ? "start" : Math.cos(angleFor(points.indexOf(p))) < -0.2 ? "end" : "middle"}
            dominantBaseline="middle"
          >
            {p.name}
          </text>
        ))}
      </svg>

      {hoveredPoint && (
        <div className="pointer-events-none absolute left-1/2 top-0 -translate-x-1/2 rounded border border-white/10 bg-[var(--surface-2)] px-2 py-1 text-xs shadow-lg">
          <span className="text-[var(--text-secondary)]">{hoveredPoint.name}: </span>
          <span className="font-medium text-white">
            {hoveredPoint.v > 0 ? "+" : ""}
            {hoveredPoint.v.toFixed(2)}σ
          </span>
        </div>
      )}

      <p className="mt-1 text-[10px] text-[var(--text-muted)]">
        Dashed ring = reference (Δ = 0). Outward = more engaged than reference, inward = less.
      </p>
    </div>
  );
}
