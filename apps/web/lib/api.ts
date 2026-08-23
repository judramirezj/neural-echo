import type { JobDetail } from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export interface CreateJobParams {
  constraintText: string;
  youtubeUrl?: string;
  file?: File;
  maxIterations: number;
  adherenceTau?: number;
  dryRun?: boolean;
}

export async function createJob(params: CreateJobParams): Promise<{ job_id: string }> {
  const form = new FormData();
  form.set("constraint_text", params.constraintText);
  if (params.youtubeUrl) form.set("youtube_url", params.youtubeUrl);
  if (params.file) form.set("file", params.file);
  form.set("dry_run", String(params.dryRun ?? false));
  form.set("max_iterations", String(params.maxIterations));
  if (params.adherenceTau !== undefined) {
    form.set("adherence_tau", String(params.adherenceTau));
  }

  const res = await fetch(`${API_URL}/jobs`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore — keep statusText
    }
    throw new Error(`Failed to create job: ${detail}`);
  }

  return res.json();
}

export async function getJob(jobId: string): Promise<JobDetail> {
  const res = await fetch(`${API_URL}/jobs/${jobId}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch job ${jobId}: ${res.statusText}`);
  }
  return res.json();
}

export function jobStreamUrl(jobId: string): string {
  return `${API_URL}/jobs/${jobId}/stream`;
}

export function artifactUrl(jobId: string, filename: string): string {
  return `${API_URL}/jobs/${jobId}/artifacts/${encodeURIComponent(filename)}`;
}
