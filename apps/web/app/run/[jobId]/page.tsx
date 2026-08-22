"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { getJob, jobStreamUrl } from "@/lib/api";
import {
  isTaggedEvent,
  type Candidate,
  type GenerationCompleteEvent,
  type JobResult,
  type JobStatusValue,
  type JobStreamMessage,
} from "@/lib/types";
import { bestByDBrain, fmtNum } from "@/lib/format";
import { StatusBanner } from "@/components/run/status-banner";
import { BestPanel } from "@/components/run/best-panel";
import { ConvergenceChart, type ConvergencePoint } from "@/components/run/convergence-chart";
import { HypothesisLog } from "@/components/run/hypothesis-log";
import { GenerationRow } from "@/components/run/generation-row";

export default function RunPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;

  const [initialLoading, setInitialLoading] = useState(true);
  const [initialError, setInitialError] = useState<string | null>(null);
  const [constraintText, setConstraintText] = useState("");
  const [status, setStatus] = useState<JobStatusValue>("pending");
  const [streamError, setStreamError] = useState<string | null>(null);
  const [doneResult, setDoneResult] = useState<JobResult | null>(null);
  const [generations, setGenerations] = useState<GenerationCompleteEvent[]>([]);
  const [staleOnLoad, setStaleOnLoad] = useState(false);

  const esRef = useRef<EventSource | null>(null);

  // Resolve current job state first — a job's SSE queue does not replay
  // history, so on a fresh page load (or refresh) we need GET /jobs/{id}
  // to know if it already reached a terminal state before deciding whether
  // to open the live stream at all.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const job = await getJob(jobId);
        if (cancelled) return;
        setConstraintText(job.constraint_text);
        setStatus(job.status);
        setStreamError(job.error);
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
          case "generation_complete":
            setGenerations((prev) => [...prev, msg]);
            break;
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
      // EventSource retries automatically; if the job is already terminal
      // this will just keep failing quietly, which is fine for a demo tool.
    };

    return () => {
      es.close();
    };
  }, [jobId, initialLoading, initialError, staleOnLoad]);

  const allCandidates = useMemo(
    () => generations.flatMap((g) => g.candidates),
    [generations]
  );

  const bestOverall: Candidate | null = useMemo(() => {
    if (doneResult?.best) return doneResult.best;
    return bestByDBrain(allCandidates);
  }, [allCandidates, doneResult]);

  const bestGenerationIndex = useMemo(() => {
    if (!bestOverall) return null;
    const found = generations.find((g) =>
      g.candidates.some(
        (c) => c.audio_path !== null && c.audio_path === bestOverall.audio_path
      )
    );
    return found?.generation_index ?? null;
  }, [generations, bestOverall]);

  const nullMedian = doneResult?.null_median ?? null;
  const noiseFloor = doneResult?.noise_floor ?? null;

  const domainMax = useMemo(() => {
    const scored = allCandidates
      .map((c) => c.D_brain)
      .filter((v): v is number => v !== null);
    const candidates = [...scored];
    if (nullMedian !== null) candidates.push(nullMedian);
    if (noiseFloor !== null) candidates.push(noiseFloor);
    const max = candidates.length > 0 ? Math.max(...candidates) : 1;
    return max * 1.15;
  }, [allCandidates, nullMedian, noiseFloor]);

  const convergencePoints: ConvergencePoint[] = useMemo(() => {
    const sorted = [...generations].sort((a, b) => a.generation_index - b.generation_index);
    const result: ConvergencePoint[] = [];
    for (const g of sorted) {
      const genMin = g.candidates.reduce<number | null>((acc, c) => {
        if (c.D_brain === null) return acc;
        if (acc === null || c.D_brain < acc) return c.D_brain;
        return acc;
      }, null);
      const prevBest = result.length > 0 ? result[result.length - 1].bestSoFar : null;
      const bestSoFar =
        genMin === null ? prevBest : prevBest === null ? genMin : Math.min(prevBest, genMin);
      result.push({
        generationIndex: g.generation_index,
        bestSoFar,
        meanDBrain: g.mean_D_brain,
      });
    }
    return result;
  }, [generations]);

  const reversedGenerations = useMemo(
    () => [...generations].sort((a, b) => b.generation_index - a.generation_index),
    [generations]
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
        <p className="rounded-md border border-[var(--status-critical)]/40 bg-[var(--status-critical)]/10 px-4 py-3 text-sm text-[var(--status-critical)]">
          {initialError}
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-10">
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-[var(--accent)]">
            Neural Echo — run {jobId}
          </p>
          <h1 className="mt-1 text-lg font-semibold text-white">
            Constraint: <span className="font-normal text-[var(--text-secondary)]">{constraintText || "—"}</span>
          </h1>
        </div>
        <StatusBanner status={status} error={streamError} />
      </header>

      {status === "done" && (
        <Link
          href={`/run/${jobId}/result`}
          className="mb-8 inline-block rounded-md bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white"
        >
          View result →
        </Link>
      )}

      {staleOnLoad && generations.length === 0 && (
        <p className="mb-8 rounded-md border border-white/10 bg-[var(--surface-1)] px-4 py-3 text-sm text-[var(--text-secondary)]">
          This job already reached a terminal state before this page connected,
          so per-generation history isn&apos;t available to replay here (the
          live event stream doesn&apos;t retain history across reconnects).
          {status === "done" ? " See the final result below." : ""}
        </p>
      )}

      {bestOverall && (
        <div className="mb-10">
          <BestPanel
            jobId={jobId}
            best={bestOverall}
            bestGenerationIndex={bestGenerationIndex ?? doneResult?.n_generations ?? 0}
            domainMax={domainMax}
            nullMedian={nullMedian}
            noiseFloor={noiseFloor}
          />
        </div>
      )}

      {convergencePoints.length > 0 && (
        <section className="mb-10 rounded-lg border border-white/10 bg-[var(--surface-1)] p-5">
          <h2 className="mb-4 text-sm font-semibold text-white">Convergence</h2>
          <ConvergenceChart points={convergencePoints} nullMedian={nullMedian} noiseFloor={noiseFloor} />
        </section>
      )}

      {generations.length > 0 && (
        <section className="mb-10 rounded-lg border border-white/10 bg-[var(--surface-1)] p-5">
          <h2 className="mb-4 text-sm font-semibold text-white">Reasoning log</h2>
          <HypothesisLog generations={generations} />
        </section>
      )}

      <div className="space-y-10">
        {reversedGenerations.map((g) => (
          <GenerationRow
            key={g.generation_index}
            jobId={jobId}
            generationIndex={g.generation_index}
            candidates={g.candidates}
            meanDBrain={g.mean_D_brain}
            elapsedS={g.elapsed_s}
            domainMax={domainMax}
            nullMedian={nullMedian}
            noiseFloor={noiseFloor}
            bestCandidateAudioPath={bestOverall?.audio_path}
          />
        ))}
      </div>

      {status === "done" && doneResult && (
        <section className="mt-10 rounded-lg border border-[var(--status-good)]/40 bg-[var(--status-good)]/5 p-5 text-sm text-[var(--text-secondary)]">
          <p className="text-white">
            Converged after {doneResult.n_generations} generation
            {doneResult.n_generations === 1 ? "" : "s"}. Noise floor {fmtNum(doneResult.noise_floor)},
            random-pair median {fmtNum(doneResult.null_median)}.
          </p>
          <Link href={`/run/${jobId}/result`} className="mt-2 inline-block text-[var(--accent)] underline">
            View the full result →
          </Link>
        </section>
      )}
    </main>
  );
}
