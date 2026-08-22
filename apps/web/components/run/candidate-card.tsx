import type { Candidate } from "@/lib/types";
import { artifactUrl } from "@/lib/api";
import { fmtNum, fmtPercent } from "@/lib/format";
import { DBrainMeter } from "./d-brain-meter";
import { Badge } from "./badge";

const REJECTED_LABEL: Record<string, string> = {
  generation_failed: "Render failed",
  near_cover: "Rejected — near-cover of reference",
  constraint_not_met: "Rejected — constraint not met",
};

interface CandidateCardProps {
  jobId: string;
  candidate: Candidate;
  domainMax: number;
  nullMedian: number | null;
  noiseFloor: number | null;
  isBest?: boolean;
}

export function CandidateCard({
  jobId,
  candidate,
  domainMax,
  nullMedian,
  noiseFloor,
  isBest,
}: CandidateCardProps) {
  const rejected = candidate.rejected_reason !== null;
  const g = candidate.genome;

  return (
    <div
      className={`rounded-md border p-3 ${
        rejected
          ? "border-white/5 bg-[var(--surface-1)]/40 opacity-60"
          : isBest
          ? "border-[var(--accent)]/60 bg-[var(--surface-1)]"
          : "border-white/10 bg-[var(--surface-1)]"
      }`}
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-white">
            {g.bpm} BPM · {g.key_mode}
          </p>
          <p className="truncate text-xs text-[var(--text-muted)]">
            {g.dynamic_arc.replace(/_/g, " ")} · {g.vocal_presence ? "vocal" : "instrumental"}
          </p>
        </div>
        {rejected ? (
          <Badge tone="critical">rejected</Badge>
        ) : (
          candidate.percentile !== null && (
            <Badge tone="muted">closer than {fmtPercent(candidate.percentile)}</Badge>
          )
        )}
      </div>

      {rejected ? (
        <p className="rounded bg-black/20 px-2 py-2 text-xs text-[var(--status-critical)]">
          {REJECTED_LABEL[candidate.rejected_reason ?? ""] ?? candidate.rejected_reason}
        </p>
      ) : (
        <>
          {candidate.audio_path && (
            <audio
              controls
              preload="none"
              src={artifactUrl(jobId, candidate.audio_path)}
              className="mb-3 h-8 w-full"
            />
          )}
          <DBrainMeter
            value={candidate.D_brain}
            domainMax={domainMax}
            nullMedian={nullMedian}
            noiseFloor={noiseFloor}
          />
        </>
      )}

      <div className="mt-3 flex flex-wrap gap-1.5">
        <Badge tone={candidate.passed_constraint ? "good" : "critical"}>
          adherence {fmtNum(candidate.adherence, 2)}
        </Badge>
        {candidate.novelty_audio_sim !== null && (
          <Badge tone={candidate.is_near_cover ? "critical" : "muted"}>
            novelty sim {fmtNum(candidate.novelty_audio_sim, 2)}
            {candidate.is_near_cover ? " · near-cover" : ""}
          </Badge>
        )}
      </div>

      <p className="mt-2 truncate text-[11px] text-[var(--text-muted)]" title={g.instrumentation.join(", ")}>
        {g.instrumentation.join(", ")}
      </p>
    </div>
  );
}
