import type { Candidate } from "@/lib/types";
import { fmtNum, fmtSeconds } from "@/lib/format";
import { CandidateCard } from "./candidate-card";

interface GenerationRowProps {
  jobId: string;
  generationIndex: number;
  candidates: Candidate[];
  meanDBrain: number | null;
  elapsedS: number;
  domainMax: number;
  nullMedian: number | null;
  noiseFloor: number | null;
  bestCandidateAudioPath: string | null | undefined;
}

export function GenerationRow({
  jobId,
  generationIndex,
  candidates,
  meanDBrain,
  elapsedS,
  domainMax,
  nullMedian,
  noiseFloor,
  bestCandidateAudioPath,
}: GenerationRowProps) {
  return (
    <section>
      <div className="mb-3 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-white">Generation {generationIndex}</h3>
        <p className="text-xs text-[var(--text-muted)]">
          mean D_brain {fmtNum(meanDBrain)} · {fmtSeconds(elapsedS)}
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {candidates.map((c, i) => (
          <CandidateCard
            key={`${generationIndex}-${i}-${c.audio_path ?? i}`}
            jobId={jobId}
            candidate={c}
            domainMax={domainMax}
            nullMedian={nullMedian}
            noiseFloor={noiseFloor}
            isBest={c.audio_path !== null && c.audio_path === bestCandidateAudioPath}
          />
        ))}
      </div>
    </section>
  );
}
