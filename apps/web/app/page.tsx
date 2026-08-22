"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { SourceInput, type SourceMode } from "@/components/setup/source-input";
import { ConstraintInput } from "@/components/setup/constraint-input";
import { AdvancedSettings } from "@/components/setup/advanced-settings";
import { createJob } from "@/lib/api";

export default function Home() {
  const router = useRouter();

  const [mode, setMode] = useState<SourceMode>("youtube");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [constraintText, setConstraintText] = useState("");
  const [maxGenerations, setMaxGenerations] = useState(6);
  const [batchSize, setBatchSize] = useState(10);
  const [adherenceTau, setAdherenceTau] = useState(0.15);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasSource = mode === "youtube" ? youtubeUrl.trim().length > 0 : file !== null;
  const canSubmit = hasSource && constraintText.trim().length > 0 && !submitting;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const { job_id } = await createJob({
        constraintText: constraintText.trim(),
        youtubeUrl: mode === "youtube" ? youtubeUrl.trim() : undefined,
        file: mode === "file" ? file ?? undefined : undefined,
        maxGenerations,
        batchSize,
        adherenceTau,
      });
      router.push(`/run/${job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start job");
      setSubmitting(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col justify-center px-6 py-16">
      <div className="mb-10">
        <p className="mb-2 text-xs font-medium uppercase tracking-widest text-[var(--accent)]">
          Neural Echo
        </p>
        <h1 className="text-2xl font-semibold text-white">
          Evolve a track toward a listener&apos;s brain response
        </h1>
        <p className="mt-2 text-sm text-[var(--text-secondary)]">
          Give it a reference track and a creative constraint. A closed-loop
          optimizer will propose, render, and score candidates against a
          brain-encoding model until it converges.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <section>
          <h2 className="mb-2 text-sm font-medium text-white">Reference track</h2>
          <SourceInput
            mode={mode}
            onModeChange={setMode}
            youtubeUrl={youtubeUrl}
            onYoutubeUrlChange={setYoutubeUrl}
            file={file}
            onFileChange={setFile}
          />
        </section>

        <section>
          <h2 className="mb-2 text-sm font-medium text-white">Creative constraint</h2>
          <ConstraintInput value={constraintText} onChange={setConstraintText} />
        </section>

        <AdvancedSettings
          maxGenerations={maxGenerations}
          onMaxGenerationsChange={setMaxGenerations}
          batchSize={batchSize}
          onBatchSizeChange={setBatchSize}
          adherenceTau={adherenceTau}
          onAdherenceTauChange={setAdherenceTau}
        />

        {error && (
          <p className="rounded-md border border-[var(--status-critical)]/40 bg-[var(--status-critical)]/10 px-3 py-2 text-sm text-[var(--status-critical)]">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={!canSubmit}
          className="w-full rounded-md bg-[var(--accent)] px-4 py-3 text-sm font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? "Starting…" : "Start evolution"}
        </button>
      </form>
    </main>
  );
}
