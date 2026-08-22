"use client";

import { useEffect, useState } from "react";

interface OEmbedData {
  title: string;
  author_name: string;
  thumbnail_url: string;
}

export type SourceMode = "youtube" | "file";

interface SourceInputProps {
  mode: SourceMode;
  onModeChange: (mode: SourceMode) => void;
  youtubeUrl: string;
  onYoutubeUrlChange: (url: string) => void;
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export function SourceInput({
  mode,
  onModeChange,
  youtubeUrl,
  onYoutubeUrlChange,
  file,
  onFileChange,
}: SourceInputProps) {
  const [oembed, setOembed] = useState<OEmbedData | null>(null);
  const [oembedError, setOembedError] = useState<string | null>(null);
  const [loadingOembed, setLoadingOembed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const handle = setTimeout(async () => {
      if (cancelled) return;
      if (mode !== "youtube" || !youtubeUrl.trim()) {
        setOembed(null);
        setOembedError(null);
        setLoadingOembed(false);
        return;
      }
      setLoadingOembed(true);
      setOembedError(null);
      try {
        const res = await fetch(
          `https://www.youtube.com/oembed?url=${encodeURIComponent(
            youtubeUrl.trim()
          )}&format=json`
        );
        if (!res.ok) throw new Error("Could not resolve this URL as a YouTube video");
        const data = (await res.json()) as OEmbedData;
        if (!cancelled) setOembed(data);
      } catch {
        if (!cancelled) {
          setOembed(null);
          setOembedError("Could not resolve this URL as a YouTube video");
        }
      } finally {
        if (!cancelled) setLoadingOembed(false);
      }
    }, 500);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [mode, youtubeUrl]);

  return (
    <div>
      <div className="mb-3 inline-flex rounded-md border border-white/10 bg-[var(--surface-1)] p-1 text-sm">
        <button
          type="button"
          onClick={() => onModeChange("youtube")}
          className={`rounded px-3 py-1.5 transition-colors ${
            mode === "youtube"
              ? "bg-[var(--accent)] text-white"
              : "text-[var(--text-secondary)] hover:text-white"
          }`}
        >
          YouTube link
        </button>
        <button
          type="button"
          onClick={() => onModeChange("file")}
          className={`rounded px-3 py-1.5 transition-colors ${
            mode === "file"
              ? "bg-[var(--accent)] text-white"
              : "text-[var(--text-secondary)] hover:text-white"
          }`}
        >
          Upload file
        </button>
      </div>

      {mode === "youtube" ? (
        <div>
          <input
            type="url"
            inputMode="url"
            placeholder="https://www.youtube.com/watch?v=..."
            value={youtubeUrl}
            onChange={(e) => onYoutubeUrlChange(e.target.value)}
            className="w-full rounded-md border border-white/10 bg-[var(--surface-1)] px-3 py-2.5 text-sm text-white placeholder:text-[var(--text-muted)] outline-none focus:border-[var(--accent)]"
          />
          {loadingOembed && (
            <p className="mt-2 text-xs text-[var(--text-muted)]">Resolving video…</p>
          )}
          {oembedError && (
            <p className="mt-2 text-xs text-[var(--status-critical)]">{oembedError}</p>
          )}
          {oembed && (
            <div className="mt-3 flex items-center gap-3 rounded-md border border-white/10 bg-[var(--surface-1)] p-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={oembed.thumbnail_url}
                alt={oembed.title}
                className="h-14 w-24 rounded object-cover"
              />
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-white">{oembed.title}</p>
                <p className="truncate text-xs text-[var(--text-muted)]">
                  {oembed.author_name}
                </p>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div>
          <label className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-white/15 bg-[var(--surface-1)] px-4 py-8 text-center transition-colors hover:border-[var(--accent)]">
            <input
              type="file"
              accept="audio/*,video/*"
              className="hidden"
              onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
            />
            {file ? (
              <div>
                <p className="text-sm font-medium text-white">{file.name}</p>
                <p className="mt-1 text-xs text-[var(--text-muted)]">
                  {(file.size / (1024 * 1024)).toFixed(1)} MB — click to replace
                </p>
              </div>
            ) : (
              <div>
                <p className="text-sm text-[var(--text-secondary)]">
                  Click to choose an audio or video file
                </p>
                <p className="mt-1 text-xs text-[var(--text-muted)]">
                  Used as the reference track for brain-response scoring
                </p>
              </div>
            )}
          </label>
        </div>
      )}
    </div>
  );
}
