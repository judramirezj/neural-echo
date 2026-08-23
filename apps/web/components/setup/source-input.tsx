"use client";

export type SourceMode = "youtube" | "file";

interface SourceInputProps {
  mode: SourceMode;
  onModeChange: (mode: SourceMode) => void;
  youtubeUrl: string;
  onYoutubeUrlChange: (url: string) => void;
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export function SourceInput({ file, onFileChange, onModeChange }: SourceInputProps) {
  return (
    <div>
      <div className="mb-4 grid grid-cols-2 rounded-xl border border-white/10 bg-black/20 p-1 text-sm">
        <button
          type="button"
          onClick={() => onModeChange("file")}
          className="rounded-lg bg-white/[0.09] px-3 py-2 font-medium text-white shadow-sm"
        >
          Upload a track
        </button>
        <button
          type="button"
          disabled
          aria-disabled="true"
          className="flex cursor-not-allowed items-center justify-center gap-2 rounded-lg px-3 py-2 text-[var(--text-muted)]"
        >
          YouTube
          <span className="rounded-full border border-violet-300/20 bg-violet-300/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-violet-200">
            Soon
          </span>
        </button>
      </div>

      <label className="group relative flex cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed border-white/15 bg-white/[0.035] px-5 py-9 text-center transition duration-300 hover:border-cyan-300/50 hover:bg-cyan-300/[0.035]">
        <input
          type="file"
          accept="audio/*,video/*"
          className="hidden"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
        <span className="mb-4 grid h-12 w-12 place-items-center rounded-2xl border border-white/10 bg-white/[0.06] text-xl text-cyan-200 transition group-hover:scale-105 group-hover:border-cyan-300/30">
          {file ? "✓" : "↑"}
        </span>
        {file ? (
          <div className="max-w-full">
            <p className="truncate text-sm font-semibold text-white">{file.name}</p>
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              {(file.size / (1024 * 1024)).toFixed(1)} MB · click to replace
            </p>
          </div>
        ) : (
          <div>
            <p className="text-sm font-semibold text-white">Choose a song you love</p>
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              MP3, WAV, M4A, FLAC, or video · we analyze the first 90 seconds
            </p>
          </div>
        )}
      </label>
    </div>
  );
}
