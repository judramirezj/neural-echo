"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { getJob, jobStreamUrl } from "@/lib/api";
import {
  isTaggedEvent,
  type IterationCompleteEvent,
  type IterationResult,
  type JobResult,
  type JobStatusValue,
  type JobStreamMessage,
} from "@/lib/types";
import { bestByScore, fmtNum } from "@/lib/format";
import { StatusBanner } from "@/components/run/status-banner";
import { BestPanel } from "@/components/run/best-panel";
import { ConvergenceChart, type ConvergencePoint } from "@/components/run/convergence-chart";
import { HypothesisLog } from "@/components/run/hypothesis-log";
import { IterationCard } from "@/components/run/iteration-card";
import { BrainResponse } from "@/components/run/brain-response";
import { OptimizationInsights } from "@/components/run/optimization-insights";

export default function RunPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;

  const [initialLoading, setInitialLoading] = useState(true);
  const [initialError, setInitialError] = useState<string | null>(null);
  const [constraintText, setConstraintText] = useState("");
  const [status, setStatus] = useState<JobStatusValue>("pending");
  const [streamError, setStreamError] = useState<string | null>(null);
  const [doneResult, setDoneResult] = useState<JobResult | null>(null);
  const [iterations, setIterations] = useState<IterationResult[]>([]);
  const [staleOnLoad, setStaleOnLoad] = useState(false);
  const [connectionInterrupted, setConnectionInterrupted] = useState(false);

  const esRef = useRef<EventSource | null>(null);

  // Resolve current state and iteration history first so refreshes can resume
  // the complete optimization story before reconnecting to live updates.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const job = await getJob(jobId);
        if (cancelled) return;
        setConstraintText(job.constraint_text);
        setStatus(job.status);
        setStreamError(job.error);
        setIterations(job.iterations ?? []);
        if (job.result) setDoneResult(job.result);
        if (job.status === "done" || job.status === "error") {
          setStaleOnLoad(true);
        }
      } catch (err) {
        if (!cancelled) {
          setInitialError(err instanceof Error ? err.message : "Failed to load job");
        }
      } finally {
        if (!cancelled) setInitialLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  useEffect(() => {
    if (initialLoading || initialError || staleOnLoad) return;

    const es = new EventSource(jobStreamUrl(jobId));
    esRef.current = es;
    es.onopen = () => setConnectionInterrupted(false);

    es.onmessage = (ev) => {
      try {
        const msg: JobStreamMessage = JSON.parse(ev.data);
        if (!isTaggedEvent(msg)) {
          setStatus(msg.status);
          setStreamError(msg.error);
          return;
        }
        switch (msg.type) {
          case "status":
            setStatus(msg.status);
            break;
          case "iteration_complete": {
            const { type, ...iteration } = msg as IterationCompleteEvent;
            void type;
            setIterations((prev) => [
              ...prev.filter((item) => item.iteration_index !== iteration.iteration_index),
              iteration,
            ]);
            break;
          }
          case "done":
            setDoneResult(msg.result);
            setStatus("done");
            es.close();
            break;
          case "error":
            setStreamError(msg.error);
            setStatus("error");
            es.close();
            break;
        }
      } catch {
        // ignore malformed frames (e.g. SSE comments) — EventSource only
        // fires onmessage for actual `data:` frames, but be defensive
      }
    };

    es.onerror = () => {
      setConnectionInterrupted(true);
    };

    return () => {
      es.close();
    };
  }, [jobId, initialLoading, initialError, staleOnLoad]);

  const bestOverall: IterationResult | null = useMemo(() => {
    if (doneResult?.best) return doneResult.best;
    return bestByScore(iterations);
  }, [iterations, doneResult]);

  const domainMax = useMemo(() => {
    const scores = iterations
      .map((it) => it.cost?.global_score ?? null)
      .filter((v): v is number => v !== null);
    const max = scores.length > 0 ? Math.max(...scores) : 1;
    return max * 1.15;
  }, [iterations]);

  const convergencePoints: ConvergencePoint[] = useMemo(() => {
    const sorted = [...iterations].sort((a, b) => a.iteration_index - b.iteration_index);
    const result: ConvergencePoint[] = [];
    let runningBest: number | null = null;
    for (const it of sorted) {
      const score = it.cost?.global_score ?? null;
      if (score !== null) {
        runningBest = runningBest === null ? score : Math.min(runningBest, score);
      }
      result.push({ iterationIndex: it.iteration_index, bestSoFar: runningBest, score });
    }
    return result;
  }, [iterations]);

  const reversedIterations = useMemo(
    () => [...iterations].sort((a, b) => b.iteration_index - a.iteration_index),
    [iterations]
  );

  if (initialLoading) {
    return (
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-16">
        <p className="text-sm text-[var(--text-muted)]">Loading job…</p>
      </main>
    );
  }

  if (initialError) {
    return (
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-16">
        <div className="rounded-2xl border border-rose-300/25 bg-rose-300/10 p-5 text-sm text-rose-100">
          <p className="font-semibold text-white">The music engine is reconnecting</p>
          <p className="mt-1 text-[var(--text-secondary)]">RunPod may still be restarting the container. Your browser can safely try again.</p>
          <div className="mt-4 flex gap-3">
            <button onClick={() => window.location.reload()} className="rounded-lg bg-white px-3 py-2 text-xs font-semibold text-[#111522]">Retry connection</button>
            <Link href="/" className="rounded-lg border border-white/15 px-3 py-2 text-xs font-semibold text-white">Start a new session</Link>
          </div>
          <p className="mt-3 font-mono text-[10px] text-rose-200/60">{initialError}</p>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-10">
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4 rounded-2xl border border-white/[0.08] bg-white/[0.025] px-5 py-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-[var(--accent)]">
            Neural Echo · Live session
          </p>
          <h1 className="mt-1 text-lg font-semibold text-white">
            Creating: <span className="font-normal text-[var(--text-secondary)]">{constraintText || "—"}</span>
          </h1>
        </div>
        <StatusBanner status={status} error={streamError} />
      </header>

      {connectionInterrupted && status !== "done" && status !== "error" && (
        <p className="mb-6 rounded-xl border border-amber-300/20 bg-amber-300/[0.08] px-4 py-3 text-xs text-amber-100">
          The live connection paused. We&apos;re retrying automatically—the optimizer can continue in the background.
        </p>
      )}

      {status === "done" && (
        <Link
          href={`/run/${jobId}/result`}
          className="neural-button mb-8 inline-block rounded-xl px-5 py-2.5 text-sm font-semibold text-white"
        >
          View result →
        </Link>
      )}

      {staleOnLoad && iterations.length === 0 && (
        <p className="mb-8 rounded-md border border-white/10 bg-[var(--surface-1)] px-4 py-3 text-sm text-[var(--text-secondary)]">
          This session was created by an older engine version, so its detailed
          iteration replay is unavailable.
          {status === "done" ? " See the final result below." : ""}
        </p>
      )}

      {bestOverall && (
        <div className="mb-10">
          <BestPanel jobId={jobId} best={bestOverall} domainMax={domainMax} />
        </div>
      )}

      {iterations.length > 0 && (
        <div className="mb-10">
          <OptimizationInsights iterations={iterations} />
        </div>
      )}

      {convergencePoints.length > 0 && (
        <section className="mb-10 rounded-lg border border-white/10 bg-[var(--surface-1)] p-5">
          <h2 className="mb-4 text-sm font-semibold text-white">Convergence</h2>
          <ConvergenceChart points={convergencePoints} />
        </section>
      )}

      <div className="mb-10">
        <BrainResponse
          jobId={jobId}
          iterationCount={iterations.length || doneResult?.n_iterations || 0}
          isLive={status === "running" || status === "preparing"}
        />
      </div>

      {iterations.length > 0 && (
        <section className="mb-10 rounded-lg border border-white/10 bg-[var(--surface-1)] p-5">
          <h2 className="mb-4 text-sm font-semibold text-white">Reasoning log</h2>
          <HypothesisLog iterations={iterations} />
        </section>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {reversedIterations.map((it) => (
          <IterationCard
            key={it.iteration_index}
            jobId={jobId}
            iteration={it}
            domainMax={domainMax}
            bestSoFar={bestOverall?.cost?.global_score ?? null}
          />
        ))}
      </div>

      {status === "done" && doneResult && (
        <section className="mt-10 rounded-lg border border-[var(--status-good)]/40 bg-[var(--status-good)]/5 p-5 text-sm text-[var(--text-secondary)]">
          <p className="text-white">
            Converged after {doneResult.n_iterations} iteration
            {doneResult.n_iterations === 1 ? "" : "s"}
            {doneResult.best?.cost ? ` — best score ${fmtNum(doneResult.best.cost.global_score)}.` : "."}
          </p>
          <Link href={`/run/${jobId}/result`} className="mt-2 inline-block text-[var(--accent)] underline">
            View the full result →
          </Link>
        </section>
      )}
    </main>
  );
}
