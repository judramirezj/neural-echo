"""Genome schema (what the optimizer's LLM emits) and the ElevenLabs client
that turns a Genome into audio. Never accepts free-form text from the LLM for
composition — everything is validated pydantic, repaired or rejected.

The LLM only ever controls chunks (text/duration/styles/context_adherence)
plus a separate top-level generation seed passed straight to ElevenLabs — no
global musical knobs (bpm, key, etc.) beyond what's expressed in prose within
each chunk's styles, matching the optimizer's single-lineage plan-mutation
loop (neural_echo/optimizer.py).
"""
import asyncio
import hashlib
import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

MAX_CHUNKS = 30
MIN_CHUNK_S = 3
MAX_CHUNK_S = 120
MIN_TOTAL_S = 3
MAX_TOTAL_S = 600


class Chunk(BaseModel):
    text: str = Field(..., description='Section label, e.g. "[Intro]", "[Drop]"')
    duration_ms: int = Field(..., ge=MIN_CHUNK_S * 1000, le=MAX_CHUNK_S * 1000)
    positive_styles: list[str] = Field(..., min_length=1, max_length=12)
    negative_styles: list[str] = Field(default_factory=list, max_length=12)
    context_adherence: str = Field(default="high")

    def to_api_dict(self) -> dict:
        return {
            "text": self.text,
            "duration_ms": self.duration_ms,
            "positive_styles": self.positive_styles,
            "negative_styles": self.negative_styles,
            "context_adherence": self.context_adherence,
        }


class Genome(BaseModel):
    """Everything the optimizer's LLM controls, serialized straight to an
    ElevenLabs Music v2 composition plan."""
    chunks: list[Chunk] = Field(..., min_length=1, max_length=MAX_CHUNKS)

    @field_validator("chunks")
    @classmethod
    def _validate_total_duration(cls, chunks: list[Chunk]) -> list[Chunk]:
        total_ms = sum(c.duration_ms for c in chunks)
        if not (MIN_TOTAL_S * 1000 <= total_ms <= MAX_TOTAL_S * 1000):
            raise ValueError(
                f"Total chunk duration {total_ms}ms outside [{MIN_TOTAL_S * 1000}, {MAX_TOTAL_S * 1000}]ms"
            )
        return chunks

    def to_composition_plan(self) -> dict:
        return {"chunks": [c.to_api_dict() for c in self.chunks]}

    def content_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def repair_genome(raw: dict) -> Genome | None:
    """Best-effort coercion of a not-quite-valid LLM genome into a valid one.
    Returns None (a hard rejection, counted as a schema failure) if it can't
    be salvaged without guessing musical content."""
    try:
        return Genome.model_validate(raw)
    except Exception as e:
        logger.warning("Genome failed validation, attempting repair: %s", e)

    try:
        chunks = raw.get("chunks", [])
        if not chunks:
            return None
        total_ms = sum(c.get("duration_ms", 0) for c in chunks)
        if total_ms > MAX_TOTAL_S * 1000:
            scale = (MAX_TOTAL_S * 1000) / total_ms
            for c in chunks:
                c["duration_ms"] = max(MIN_CHUNK_S * 1000, int(c["duration_ms"] * scale))
        elif total_ms < MIN_TOTAL_S * 1000 and chunks:
            chunks[-1]["duration_ms"] += (MIN_TOTAL_S * 1000 - total_ms)
        raw["chunks"] = chunks
        return Genome.model_validate(raw)
    except Exception as e:
        logger.error("Genome repair failed, rejecting: %s", e)
        return None


class GenerationResult(BaseModel):
    genome: Genome
    audio_path: str
    dry_run: bool
    error: str | None = None


class ElevenLabsGenerator:
    """Async batch generator with a semaphore (brief §4 throughput) and a
    --dry-run stub mode that returns pre-rendered clips at zero API cost, so
    the optimizer loop and UI can be developed/tested without spending
    ElevenLabs credits on every iteration.
    """

    def __init__(
        self,
        output_dir: Path,
        dry_run: bool = False,
        stub_clips_dir: Path | None = None,
        max_concurrency: int = 2,  # ElevenLabs free/starter tiers cap at 2 concurrent requests
        api_key: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.stub_clips_dir = Path(stub_clips_dir) if stub_clips_dir else None
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = None
        if not dry_run:
            from elevenlabs.client import AsyncElevenLabs
            self._client = AsyncElevenLabs(api_key=api_key or os.environ["ELEVENLABS_API_KEY"])

    def _stub_clip_for(self, genome: Genome) -> Path:
        if not self.stub_clips_dir:
            raise RuntimeError("dry_run=True requires stub_clips_dir")
        stubs = sorted(self.stub_clips_dir.glob("*.mp3"))
        if not stubs:
            raise RuntimeError(f"No stub clips found in {self.stub_clips_dir}")
        idx = int(genome.content_hash(), 16) % len(stubs)
        return stubs[idx]

    async def generate_one(self, genome: Genome, seed: int | None = None) -> GenerationResult:
        # Cache key includes the seed: the same plan rendered with a different
        # seed is different audio, so content_hash() alone would wrongly serve
        # a stale cached file for a reseeded retry of the same plan.
        cache_key = f"{genome.content_hash()}_{seed}" if seed is not None else genome.content_hash()
        out_path = self.output_dir / f"{cache_key}.mp3"
        if out_path.exists():
            return GenerationResult(genome=genome, audio_path=str(out_path), dry_run=self.dry_run)

        if self.dry_run:
            import shutil
            src = self._stub_clip_for(genome)
            shutil.copy(src, out_path)
            return GenerationResult(genome=genome, audio_path=str(out_path), dry_run=True)

        async with self._semaphore:
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    compose_kwargs = {
                        "composition_plan": genome.to_composition_plan(),
                        "model_id": "music_v2",
                        "respect_sections_durations": True,
                    }
                    if seed is not None:
                        compose_kwargs["seed"] = seed
                    chunks = []
                    async for b in self._client.music.compose(**compose_kwargs):
                        chunks.append(b)
                    out_path.write_bytes(b"".join(chunks))
                    return GenerationResult(genome=genome, audio_path=str(out_path), dry_run=False)
                except Exception as e:
                    msg = str(e)
                    is_rate_limit = "429" in msg or "concurrent_limit_exceeded" in msg
                    is_transient = (
                        "status_code: 5" in msg or "internal_server_error" in msg
                        or "timeout" in msg.lower() or "connection" in msg.lower()
                    )
                    if (is_rate_limit or is_transient) and attempt < max_attempts - 1:
                        delay = 2.0 * (2 ** attempt)
                        logger.warning(
                            "Transient error generating genome %s, retrying in %.1fs (attempt %d/%d): %s",
                            cache_key, delay, attempt + 1, max_attempts, msg,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # Non-transient (e.g. ToS/copyright rejection): return the error
                    # immediately without exhausting the retry budget — the caller
                    # (the optimizer) is responsible for asking the LLM to revise
                    # content-policy violations, which retrying alone can't fix.
                    logger.error("ElevenLabs generation failed for genome %s: %s", cache_key, e)
                    return GenerationResult(genome=genome, audio_path="", dry_run=False, error=msg)

    async def generate_batch(self, genomes: list[Genome]) -> list[GenerationResult]:
        return await asyncio.gather(*(self.generate_one(g) for g in genomes))
