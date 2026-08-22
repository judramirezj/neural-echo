import type { JobStatusValue } from "@/lib/types";

interface StatusBannerProps {
  status: JobStatusValue;
  error: string | null;
}

const LABELS: Record<JobStatusValue, string> = {
  pending: "Queued…",
  preparing: "Preparing reference track and calibration…",
  running: "Running optimizer…",
  done: "Converged.",
  error: "Job failed.",
};

export function StatusBanner({ status, error }: StatusBannerProps) {
  if (status === "error") {
    return (
      <div className="rounded-md border border-[var(--status-critical)]/40 bg-[var(--status-critical)]/10 px-4 py-3 text-sm text-[var(--status-critical)]">
        <p className="font-medium">Job failed</p>
        <p className="mt-1">{error ?? "Unknown error"}</p>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
      {(status === "pending" || status === "preparing" || status === "running") && (
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--accent)] opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--accent)]" />
        </span>
      )}
      <span>{LABELS[status]}</span>
    </div>
  );
}
