// Types mirror services/api/main.py and services/api/jobs.py exactly.
// Where the written API spec disagreed with the actual backend source
// (neural_echo/generator.py), the source wins — see Chunk.context_adherence.

export type DynamicArc = "flat" | "crescendo" | "peak_and_fall" | "multi_peak";

export interface Chunk {
  text: string;
  duration_ms: number;
  positive_styles: string[];
  negative_styles: string[];
  // Backend (neural_echo/generator.py) defines this as a free-form string
  // (e.g. "high"), not a float as an earlier draft of the API contract
  // suggested. Rendered as text, not a numeric bar.
  context_adherence: string;
}

export interface Genome {
  bpm: number;
  key_mode: string;
  instrumentation: string[];
  texture_density: number;
  dynamic_arc: DynamicArc;
  vocal_presence: boolean;
  brightness: number;
  space_reverb: number;
  section_count: number;
  chunks: Chunk[];
  rationale: string;
}

export type RejectedReason =
  | "generation_failed"
  | "near_cover"
  | "constraint_not_met"
  | null;

export interface Candidate {
  genome: Genome;
  audio_path: string | null;
  D_brain: number | null;
  percentile: number | null;
  d_spatial: number | null;
  d_dynamics: number | null;
  d_geometry: number | null;
  adherence: number | null;
  novelty_audio_sim: number | null;
  is_near_cover: boolean | null;
  passed_constraint: boolean;
  rejected_reason: RejectedReason;
  per_network_deltas: Record<string, number>;
}

export type JobStatusValue = "pending" | "preparing" | "running" | "done" | "error";

export interface JobResult {
  best: Candidate | null;
  n_generations: number;
  noise_floor: number;
  null_median: number;
  reference_analysis: Record<string, unknown>;
}

export interface JobStatusDict {
  id: string;
  status: JobStatusValue;
  error: string | null;
  n_generations: number;
  constraint_text: string;
  dry_run: boolean;
}

export interface JobDetail extends JobStatusDict {
  result: JobResult | null;
}

export interface GenerationCompleteEvent {
  type: "generation_complete";
  generation_index: number;
  hypothesis: string;
  learned_insights: string;
  candidates: Candidate[];
  best: Candidate | null;
  mean_D_brain: number | null;
  elapsed_s: number;
}

export interface StatusEvent {
  type: "status";
  status: "preparing" | "running";
}

export interface DoneEvent {
  type: "done";
  result: JobResult;
}

export interface ErrorEvent {
  type: "error";
  error: string;
}

// The very first message the stream sends is job.to_status_dict() with no
// "type" field at all (see services/api/jobs.py::stream_job) — it's a plain
// JobStatusDict. Every subsequent message is a tagged JobStreamEvent.
export type JobStreamMessage =
  | JobStatusDict
  | StatusEvent
  | GenerationCompleteEvent
  | DoneEvent
  | ErrorEvent;

export function isTaggedEvent(
  msg: JobStreamMessage
): msg is StatusEvent | GenerationCompleteEvent | DoneEvent | ErrorEvent {
  return "type" in msg;
}
