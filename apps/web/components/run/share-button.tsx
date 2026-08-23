"use client";

import { useState } from "react";

export function ShareButton() {
  const [label, setLabel] = useState("Share result");

  async function share() {
    const data = {
      title: "My Neural Echo",
      text: "Listen to the music I evolved from a favorite memory.",
      url: window.location.href,
    };
    try {
      if (navigator.share) {
        await navigator.share(data);
        return;
      }
      await navigator.clipboard.writeText(data.url);
      setLabel("Link copied");
      window.setTimeout(() => setLabel("Share result"), 1800);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setLabel("Couldn't share");
      window.setTimeout(() => setLabel("Share result"), 1800);
    }
  }

  return (
    <button
      type="button"
      onClick={share}
      className="rounded-xl border border-white/15 bg-white/[0.04] px-4 py-2 text-sm font-semibold text-white transition hover:border-violet-300/35 hover:bg-violet-300/[0.08]"
    >
      {label}
    </button>
  );
}
