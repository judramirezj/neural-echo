export function SiteFooter() {
  return (
    <footer className="border-t border-white/10 bg-[var(--surface-1)] px-6 py-4 text-xs leading-relaxed text-[var(--text-muted)]">
      <div className="mx-auto max-w-6xl space-y-1">
        <p>
          Research demo — TRIBE v2 is CC-BY-NC-4.0, non-commercial use only.
          Brain-response scores are a research proxy, not a clinical or
          validated measurement.
        </p>
        <p>
          When a reference track is sourced from a YouTube URL, this demo
          downloads its audio for scoring purposes only — that is a
          ToS-grey-area affordance for demonstration, not a production data
          path. Uploading your own file avoids it entirely.
        </p>
      </div>
    </footer>
  );
}
