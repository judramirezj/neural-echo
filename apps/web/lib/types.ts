// Types mirror services/api/main.py and services/api/jobs.py exactly.

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
  chunks: Chunk[];
}

export interface RegionScore {
  region: string;
  distance: number;
  arc_correlation: number;
  score: number;
}

export interface WindowSummary {
  window_index: number;
  rms_error: number;
  mean_bias: number;
}

export interface WorstCell {
  window_index: number;
  region: string;
  candidate: number;
  target: number;
  difference: number;
}

export interface CostResult {
  global_score: number; // lower is better
  regions: RegionScore[];
  windows: WindowSummary[];
  worst_cell: WorstCell;
  laterality: Record<string, number>;
}

export type RejectedReason =
  | "generation_failed"
  | "near_cover"
  | "constraint_not_met"
  | null;

export interface IterationResult {
  iteration_index: number;
  reasoning: string;
  changes_summary: string;
  plan: Genome;
  seed: number;
  audio_path: string | null;
  is_best: boolean;
  elapsed_s: number;
  cost: CostResult | null;
  rejected_reason: RejectedReason;
  adherence: number | null;
  novelty_audio_sim: number | null;
  is_near_cover: boolean | null;
}

export type JobStatusValue = "pending" | "preparing" | "running" | "done" | "error";

export interface JobResult {
  best: IterationResult | null;
  n_iterations: number;
  reference_analysis: Record<string, unknown>;
}

export interface JobStatusDict {
  id: string;
  status: JobStatusValue;
  error: string | null;
  n_iterations: number;
  constraint_text: string;
  dry_run: boolean;
}

export interface JobDetail extends JobStatusDict {
  result: JobResult | null;
}

export interface BrainFrameSummary {
  iteration_index: number;
  mean_mismatch: number;
  peak_mismatch: number;
  active_fraction: number;
}

export interface BrainVisualizationResponse {
  // Plotly's JSON schema is deliberately passed through from Python. The
  // renderer owns validation; keeping it unknown here avoids mirroring the
  // enormous and versioned Plotly schema in our API contract.
  figure: {
    data: unknown[];
    layout: Record<string, unknown>;
    frames: unknown[];
  };
  meta: {
    frames: BrainFrameSummary[];
    threshold: number;
    scale_max: number;
    latest_iteration: number;
  };
}

export interface IterationCompleteEvent extends IterationResult {
  type: "iteration_complete";
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
  | IterationCompleteEvent
  | DoneEvent
  | ErrorEvent;

export function isTaggedEvent(
  msg: JobStreamMessage
): msg is StatusEvent | IterationCompleteEvent | DoneEvent | ErrorEvent {
  return "type" in msg;
}
