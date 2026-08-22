"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { getJob, artifactUrl } from "@/lib/api";
import type { JobDetail } from "@/lib/types";
import { fmtNum, fmtPercent } from "@/lib/format";

export default function ResultPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;

  const [job, setJob] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await getJob(jobId);
        if (!cancelled) setJob(data);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load job");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (loading) {
    return (
      <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-16">
        <p className="text-sm text-[var(--text-muted)]">Loading result…</p>
      </main>
    );
  }

  if (error || !job) {
    return (
      <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-16">
        <p className="rounded-md border border-[var(--status-critical)]/40 bg-[var(--status-critical)]/10 px-4 py-3 text-sm text-[var(--status-critical)]">
          {error ?? "Job not found"}
        </p>
      </main>
    );
  }

  if (job.status !== "done") {
    return (
      <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-16">
        <p className="mb-4 text-sm text-[var(--text-secondary)]">
          This job hasn&apos;t finished yet (status: {job.status}).
        </p>
        <Link href={`/run/${jobId}`} className="text-sm text-[var(--accent)] underline">
          Back to the live run →
        </Link>
      </main>
    );
  }

  const result = job.result;
  const best = result?.best ?? null;

  if (!result || !best) {
    return (
      <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-16">
        <p className="text-sm text-[var(--text-secondary)]">
          This job finished without producing a scored candidate — every
          candidate across all generations was rejected before scoring.
        </p>
      </main>
    );
  }

  const progressPct =
    best.D_brain !== null && result.null_median > result.noise_floor
      ? Math.min(
          100,
          Math.max(
            0,
            ((result.null_median - best.D_brain) /
              (result.null_median - result.noise_floor)) *
              100
          )
        )
      : null;

  const summary = buildSummary(best.percentile, progressPct);

  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-6 py-12">
      <p className="mb-1 text-xs font-medium uppercase tracking-widest text-[var(--accent)]">
        Result
      </p>
      <h1 className="mb-6 text-2xl font-semibold text-white">Winning candidate</h1>

      <div className="mb-6 rounded-lg border border-[var(--accent)]/50 bg-[var(--surface-1)] p-5">
        {best.audio_path && (
          <audio controls src={artifactUrl(jobId, best.audio_path)} className="mb-4 w-full" />
        )}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex gap-6 text-sm">
            <Metric label="D_brain" value={fmtNum(best.D_brain)} />
            <Metric label="Percentile" value={fmtPercent(best.percentile)} />
            <Metric label="Adherence" value={fmtNum(best.adherence, 2)} />
            <Metric label="Novelty sim" value={fmtNum(best.novelty_audio_sim, 2)} />
          </div>
          {best.audio_path && (
            <a
              href={artifactUrl(jobId, best.audio_path)}
              download
              className="rounded-md bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white"
            >
              Download MP3
            </a>
          )}
        </div>
      </div>

      <p className="mb-8 rounded-md border border-white/10 bg-[var(--surface-1)] px-4 py-3 text-sm leading-relaxed text-[var(--text-secondary)]">
        {summary}
      </p>

      <section className="mb-8 rounded-lg border border-white/10 bg-[var(--surface-1)] p-5">
        <h2 className="mb-3 text-sm font-semibold text-white">Run summary</h2>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
          <Metric label="Generations run" value={String(result.n_generations)} />
          <Metric label="Noise floor" value={fmtNum(result.noise_floor)} />
          <Metric label="Random-pair median" value={fmtNum(result.null_median)} />
          <Metric label="Constraint" value={job.constraint_text || "—"} />
        </div>
      </section>

      <details className="mb-8 rounded-lg border border-white/10 bg-[var(--surface-1)] p-5">
        <summary className="cursor-pointer text-sm font-semibold text-white">
          Composition plan (genome JSON)
        </summary>
        <pre className="mt-4 max-h-[480px] overflow-auto rounded bg-black/30 p-3 text-xs text-[var(--text-secondary)]">
          {JSON.stringify(best.genome, null, 2)}
        </pre>
      </details>

      <Link href={`/run/${jobId}`} className="text-sm text-[var(--accent)] underline">
        ← Back to the evolution log
      </Link>
    </main>
  );
}

function buildSummary(percentile: number | null, progressPct: number | null): string {
  const parts: string[] = [];
  if (percentile !== null) {
    parts.push(
      `This track's brain-response match is closer to the reference than ${percentile.toFixed(
        0
      )}% of random music pairs.`
    );
  }
  if (progressPct !== null) {
    parts.push(
      `It closed ${progressPct.toFixed(
        0
      )}% of the gap between "random music" and the theoretical noise floor for this reference.`
    );
  }
  if (parts.length === 0) {
    return "Scoring diagnostics for this candidate are incomplete.";
  }
  return parts.join(" ");
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-[var(--text-muted)]">{label}</p>
      <p className="tabular-nums text-sm font-medium text-white">{value}</p>
    </div>
  );
}
