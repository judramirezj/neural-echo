"use client";

import Image from "next/image";
import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import projectLogo from "../../../project-logo.png";
import { SourceInput, type SourceMode } from "@/components/setup/source-input";
import { ConstraintInput } from "@/components/setup/constraint-input";
import { AdvancedSettings } from "@/components/setup/advanced-settings";
import { createJob } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [mode, setMode] = useState<SourceMode>("file");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [constraintText, setConstraintText] = useState("");
  const [maxIterations, setMaxIterations] = useState(10);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = file !== null && constraintText.trim().length > 0 && !submitting;

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const { job_id } = await createJob({
        constraintText: constraintText.trim(),
        file: file ?? undefined,
        maxIterations,
      });
      router.push(`/run/${job_id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "We couldn't start your session.");
      setSubmitting(false);
    }
  }

  return (
    <main className="relative isolate flex flex-1 overflow-hidden">
      <div className="neural-aurora pointer-events-none absolute inset-0" aria-hidden="true" />
      <div className="mx-auto grid w-full max-w-7xl items-center gap-12 px-6 py-12 lg:grid-cols-[1.05fr_0.95fr] lg:px-10 lg:py-16">
        <section className="relative z-10 max-w-2xl">
          <div className="mb-8 flex items-center gap-3">
            <div className="relative h-14 w-44 overflow-hidden rounded-xl border border-white/10 bg-[#111] shadow-[0_0_28px_rgba(103,232,249,0.18)]">
              <Image src={projectLogo} alt="Neural Echo — Neural Music Synthesizer" fill sizes="176px" className="object-cover" priority />
            </div>
            <p className="hidden text-[10px] uppercase tracking-[0.2em] text-cyan-200/60 sm:block">Music, remembered differently</p>
          </div>

          <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-200/15 bg-cyan-200/[0.06] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-100">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-300 shadow-[0_0_10px_#67e8f9]" />
            Neural music synthesizer
          </p>
          <h1 className="max-w-2xl text-4xl font-semibold leading-[1.02] tracking-[-0.045em] text-white sm:text-6xl">
            Turn the feeling of a favorite song into
            <span className="neural-gradient-text"> something entirely yours.</span>
          </h1>
          <p className="mt-6 max-w-xl text-base leading-relaxed text-[var(--text-secondary)] sm:text-lg">
            Upload a track tied to a memory. Neural Echo creates new music, listens to its predicted
            brain response, and keeps evolving until the feeling moves closer.
          </p>

          <div className="mt-8 grid max-w-xl grid-cols-3 gap-3">
            <TrustStat value="90 sec" label="memory sample" />
            <TrustStat value="20k+" label="cortical points" />
            <TrustStat value="Live" label="evolution replay" />
          </div>
        </section>

        <section className="relative z-10">
          <div className="neural-orbit pointer-events-none absolute -inset-16 hidden lg:block" aria-hidden="true">
            <span className="neural-node neural-node-a" />
            <span className="neural-node neural-node-b" />
            <span className="neural-node neural-node-c" />
          </div>
          <form
            onSubmit={handleSubmit}
            className="relative space-y-6 rounded-[28px] border border-white/[0.12] bg-[#111522]/85 p-5 shadow-[0_35px_120px_rgba(0,0,0,0.55),0_0_80px_rgba(80,120,255,0.08)] backdrop-blur-xl sm:p-7"
          >
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-violet-200/70">Create your echo</p>
              <h2 className="mt-1 text-xl font-semibold tracking-tight text-white">Start with a memory</h2>
              <p className="mt-1 text-xs leading-relaxed text-[var(--text-muted)]">Your upload is the emotional reference—not a song to copy.</p>
            </div>

            <section>
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/80">1 · Reference song</label>
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
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/80">2 · Make it yours</label>
              <ConstraintInput value={constraintText} onChange={setConstraintText} />
            </section>

            <AdvancedSettings maxIterations={maxIterations} onMaxIterationsChange={setMaxIterations} />

            {error && (
              <div className="rounded-xl border border-rose-300/25 bg-rose-300/10 px-4 py-3 text-sm text-rose-100">
                <p>{friendlyStartError(error)}</p>
                <button type="button" onClick={() => setError(null)} className="mt-2 text-xs font-semibold underline underline-offset-4">Try again</button>
              </div>
            )}

            <button
              type="submit"
              disabled={!canSubmit}
              className="neural-button group relative w-full overflow-hidden rounded-xl px-5 py-3.5 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-35"
            >
              <span className="relative z-10">{submitting ? "Opening the memory…" : "Generate my neural echo"}</span>
            </button>
            <p className="text-center text-[10px] text-[var(--text-muted)]">Each run creates original audio through an iterative brain-response model.</p>
          </form>
        </section>
      </div>
    </main>
  );
}

function TrustStat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-xl border border-white/[0.08] bg-white/[0.035] px-3 py-3 backdrop-blur">
      <p className="font-mono text-sm font-semibold text-white">{value}</p>
      <p className="mt-0.5 text-[10px] text-[var(--text-muted)]">{label}</p>
    </div>
  );
}

function friendlyStartError(error: string): string {
  const message = error.toLowerCase();
  if (message.includes("fetch") || message.includes("network")) {
    return "The music engine is reconnecting. Give it a moment, then try again.";
  }
  if (message.includes("413") || message.includes("too large")) {
    return "That file is too large. Try a shorter audio file under 100 MB.";
  }
  return "We couldn't start this session. Please check the file and try once more.";
}
