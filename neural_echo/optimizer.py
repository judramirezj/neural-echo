"""The closed-loop optimizer (brief §4): LLM proposes a batch of genomes ->
ElevenLabs renders them -> TRIBE scores them against the reference -> the LLM
sees full diagnostics (including CLAP adherence/novelty rejections) and
writes a hypothesis + the next batch. Runs until budget/generation cap,
plateau, or within 15% of the noise floor.

Every generation's LLM hypothesis is captured as a first-class field on
GenerationResult specifically so callers (the API's SSE stream, the UI) can
surface "the reasoning log" live — per the product brief this is what makes
the optimizer legible rather than a black box, and it is never dropped or
summarized away between here and the frontend.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from . import analysis, calibration, compat
from .generator import ElevenLabsGenerator, Genome, repair_genome

logger = logging.getLogger(__name__)

MODEL_ID = "claude-sonnet-5"
MAX_GENERATIONS_DEFAULT = 6
BATCH_SIZE_DEFAULT = 10
PLATEAU_GENERATIONS = 2
FLOOR_PROXIMITY_STOP = 1.15  # stop once within 15% of (D_brain - floor) -> floor


GENOME_JSON_SCHEMA = Genome.model_json_schema()
# Claude tool schemas are self-contained; inline pydantic's $defs so the
# schema doesn't reference external $refs the API can't resolve.
if "$defs" in GENOME_JSON_SCHEMA:
    _defs = GENOME_JSON_SCHEMA.pop("$defs")

    def _inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _inline(_defs[ref_name])
            return {k: _inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_inline(v) for v in node]
        return node

    GENOME_JSON_SCHEMA = _inline(GENOME_JSON_SCHEMA)

PROPOSE_GENOMES_TOOL = {
    "name": "propose_genomes",
    "description": "Propose a batch of song genomes (ElevenLabs Music v2 composition plans plus global knobs).",
    "input_schema": {
        "type": "object",
        "properties": {
            "hypothesis": {
                "type": "string",
                "description": "1-3 sentences: what you believe is driving the scores so far, and why this batch is designed the way it is. Written for a human watching the run live.",
            },
            "learned_insights": {
                "type": "string",
                "description": "Persistent notes carried forward to the next generation — accumulate, don't just repeat.",
            },
            "genomes": {
                "type": "array",
                "items": GENOME_JSON_SCHEMA,
            },
        },
        "required": ["hypothesis", "learned_insights", "genomes"],
    },
}


@dataclass
class CandidateResult:
    genome: Genome
    audio_path: str
    D_brain: float | None = None
    percentile: float | None = None
    d_spatial: float | None = None
    d_dynamics: float | None = None
    d_geometry: float | None = None
    adherence: float | None = None
    novelty_audio_sim: float | None = None
    is_near_cover: bool | None = None
    passed_constraint: bool = False
    rejected_reason: str | None = None
    per_network_deltas: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    generation_index: int
    hypothesis: str
    learned_insights: str
    candidates: list[CandidateResult]
    best: CandidateResult | None
    mean_D_brain: float | None
    elapsed_s: float


class OptimizerRun:
    """One end-to-end optimization run against a single reference clip."""

    def __init__(
        self,
        reference_audio_path: str,
        constraint_text: str,
        bundle: calibration.CalibrationBundle,
        db_path: Path,
        dry_run: bool = False,
        stub_clips_dir: Path | None = None,
        batch_size: int = BATCH_SIZE_DEFAULT,
        max_generations: int = MAX_GENERATIONS_DEFAULT,
        adherence_tau: float = 0.15,
        anthropic_client: anthropic.Anthropic | None = None,
        on_generation=None,  # optional callback(GenerationResult) for live SSE streaming
    ):
        self.reference_audio_path = reference_audio_path
        self.constraint_text = constraint_text
        self.bundle = bundle
        self.batch_size = batch_size
        self.max_generations = max_generations
        self.adherence_tau = adherence_tau
        self.on_generation = on_generation

        self.model = compat.get_tribe_model()
        self.client = anthropic_client or anthropic.Anthropic()
        self.generator = ElevenLabsGenerator(
            output_dir=Path("data/generated"), dry_run=dry_run, stub_clips_dir=stub_clips_dir,
        )

        self.db_path = db_path
        self._init_db()

        self.reference_analysis = analysis.analyze_reference(reference_audio_path)
        # Computed ONCE and reused for every candidate in every generation —
        # see FINDINGS.md §8: the naive per-candidate helper recomputes the
        # reference's TRIBE pass every time, doubling cost for nothing.
        self.reference_profile = calibration.compute_profile_for_clip(
            self.model, bundle, Path(reference_audio_path)
        )
        self.reference_clap_embedding = analysis.clap_audio_embedding(reference_audio_path)

        self.history: list[GenerationResult] = []
        self.learned_insights = ""
        self.elite: list[CandidateResult] = []

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_index INTEGER,
                hypothesis TEXT,
                learned_insights TEXT,
                created_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_index INTEGER,
                audio_hash TEXT,
                genome_json TEXT,
                audio_path TEXT,
                D_brain REAL,
                percentile REAL,
                d_spatial REAL,
                d_dynamics REAL,
                d_geometry REAL,
                adherence REAL,
                novelty_audio_sim REAL,
                is_near_cover INTEGER,
                passed_constraint INTEGER,
                rejected_reason TEXT,
                created_at REAL
            )
        """)
        conn.commit()
        conn.close()

    def _log_generation(self, gen: GenerationResult):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO generations (generation_index, hypothesis, learned_insights, created_at) VALUES (?,?,?,?)",
            (gen.generation_index, gen.hypothesis, gen.learned_insights, time.time()),
        )
        for c in gen.candidates:
            conn.execute(
                """INSERT INTO candidates
                (generation_index, audio_hash, genome_json, audio_path, D_brain, percentile,
                 d_spatial, d_dynamics, d_geometry, adherence, novelty_audio_sim, is_near_cover,
                 passed_constraint, rejected_reason, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    gen.generation_index, c.genome.content_hash(), c.genome.model_dump_json(), c.audio_path,
                    c.D_brain, c.percentile, c.d_spatial, c.d_dynamics, c.d_geometry, c.adherence,
                    c.novelty_audio_sim, int(bool(c.is_near_cover)), int(c.passed_constraint),
                    c.rejected_reason, time.time(),
                ),
            )
        conn.commit()
        conn.close()

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict:
        response = self.client.messages.create(
            model=MODEL_ID,
            max_tokens=16000,
            system=system_prompt,
            tools=[PROPOSE_GENOMES_TOOL],
            tool_choice={"type": "tool", "name": "propose_genomes"},
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "propose_genomes":
                return block.input
        raise RuntimeError("LLM did not return a propose_genomes tool call")

    def _score_candidate(self, genome: Genome, audio_path: str) -> CandidateResult:
        result = CandidateResult(genome=genome, audio_path=audio_path)
        if not audio_path:
            result.rejected_reason = "generation_failed"
            return result

        # Hard filters first (brief §5) — never blended into D_brain.
        novelty = analysis.novelty_check(
            audio_path, self.reference_audio_path, self.reference_analysis,
            reference_embedding=self.reference_clap_embedding,
        )
        result.novelty_audio_sim = novelty["audio_similarity"]
        result.is_near_cover = novelty["is_near_cover"]
        if novelty["is_near_cover"]:
            result.rejected_reason = "near_cover"
            return result

        adherence = analysis.constraint_adherence(audio_path, self.constraint_text)
        result.adherence = adherence
        if adherence < self.adherence_tau:
            result.rejected_reason = "constraint_not_met"
            return result

        result.passed_constraint = True
        # reference_profile computed once in __init__ and reused here — see
        # FINDINGS.md §8, this used to recompute the reference's TRIBE pass
        # on every single candidate.
        cand_profile = calibration.compute_profile_for_clip(self.model, self.bundle, Path(audio_path))
        dist = calibration.score_candidate_against_profile(self.bundle, self.reference_profile, cand_profile)
        result.D_brain = dist.D_brain
        result.percentile = dist.percentile
        result.d_spatial = dist.d_spatial
        result.d_dynamics = dist.d_dynamics
        result.d_geometry = dist.d_geometry

        from . import metric as metric_mod
        network_labels_masked = (
            self.bundle.network_labels[self.bundle.vertex_mask] if self.bundle.network_labels is not None else None
        )
        result.per_network_deltas = metric_mod.per_network_deltas(cand_profile, self.reference_profile, network_labels_masked)
        return result

    def _build_gen0_prompt(self) -> tuple[str, str]:
        system = (
            "You are the creative director of Neural Echo, a system that composes original songs "
            "engineered to evoke a similar predicted brain response (via Meta's TRIBE v2 brain-encoding "
            "model) to a reference track, while satisfying a user's creative constraint and remaining "
            "clearly original — never a clone of the reference. You propose ElevenLabs Music v2 "
            "composition plans as structured genomes. You will iterate over several generations; explain "
            "your reasoning in `hypothesis` every time, in plain language a non-expert user watching this "
            "live can follow — that reasoning log is the point of this product."
        )
        user = {
            "task": "Generation 0: propose maximally diverse genomes.",
            "reference_track_analysis": self.reference_analysis,
            "user_constraint": self.constraint_text,
            "instructions": (
                f"Propose exactly {self.batch_size} genomes with a wide spread across BPM, dynamic_arc, "
                "and instrumentation — do not converge on one style yet. Every genome must independently "
                "satisfy the user_constraint and must NOT closely imitate the reference's exact "
                "instrumentation+tempo+key combination (that would be flagged as a near-cover and rejected)."
            ),
        }
        return system, json.dumps(user, indent=2)

    def _build_gen_n_prompt(self) -> tuple[str, str]:
        system = (
            "You are the creative director of Neural Echo (see prior context: composing songs that "
            "evoke a similar brain response to a reference while satisfying a user constraint and staying "
            "original). Use the run history and per-network diagnostics below to reason about what is "
            "moving the score, then propose the next batch."
        )
        history_table = []
        for gen in self.history:
            for c in gen.candidates:
                history_table.append({
                    "generation": gen.generation_index,
                    "bpm": c.genome.bpm, "dynamic_arc": c.genome.dynamic_arc.value,
                    "instrumentation": c.genome.instrumentation,
                    "D_brain": c.D_brain, "percentile": c.percentile,
                    "d_spatial": c.d_spatial, "d_dynamics": c.d_dynamics, "d_geometry": c.d_geometry,
                    "adherence": c.adherence, "novelty_audio_sim": c.novelty_audio_sim,
                    "passed_constraint": c.passed_constraint, "rejected_reason": c.rejected_reason,
                })

        last_gen = self.history[-1]
        scored = [c for c in last_gen.candidates if c.D_brain is not None]
        best = min(scored, key=lambda c: c.D_brain) if scored else None
        worst = max(scored, key=lambda c: c.D_brain) if scored else None

        user = {
            "user_constraint": self.constraint_text,
            "noise_floor": self.bundle.floor,
            "null_median": float(self.bundle.null_distribution.mean()),
            "history": history_table,
            "best_candidate_per_network_deltas_sigma": best.per_network_deltas if best else None,
            "worst_candidate_per_network_deltas_sigma": worst.per_network_deltas if worst else None,
            "learned_insights_so_far": self.learned_insights,
            "instructions": (
                f"Write a short hypothesis about what is driving the score. Then propose exactly "
                f"{self.batch_size} new genomes: roughly 6 exploiting the best-scoring region found so "
                "far (small mutations to bpm/instrumentation/dynamic_arc around the best candidate), and "
                "4 exploring new regions of the space. Every genome must satisfy the user_constraint and "
                "avoid the near-cover rejection. Lower D_brain is better; getting closer to noise_floor is "
                "the goal. Update learned_insights — carry forward what you've learned, don't just repeat it."
            ),
        }
        return system, json.dumps(user, indent=2)

    def _should_stop(self) -> str | None:
        if len(self.history) >= self.max_generations:
            return "max_generations"
        if len(self.history) >= PLATEAU_GENERATIONS + 1:
            recent_bests = []
            for gen in self.history[-(PLATEAU_GENERATIONS + 1):]:
                scored = [c.D_brain for c in gen.candidates if c.D_brain is not None]
                if scored:
                    recent_bests.append(min(scored))
            if len(recent_bests) == PLATEAU_GENERATIONS + 1 and min(recent_bests[:-1]) <= recent_bests[-1]:
                return "plateau"
        last_gen = self.history[-1] if self.history else None
        if last_gen:
            scored = [c.D_brain for c in last_gen.candidates if c.D_brain is not None]
            if scored:
                best = min(scored)
                span = float(self.bundle.null_distribution.mean()) - self.bundle.floor
                if span > 1e-9 and (best - self.bundle.floor) <= FLOOR_PROXIMITY_STOP * 0.15 * span:
                    return "near_floor"
        return None

    async def run(self) -> list[GenerationResult]:
        while True:
            gen_idx = len(self.history)
            t0 = time.time()

            if gen_idx == 0:
                system, user = self._build_gen0_prompt()
            else:
                system, user = self._build_gen_n_prompt()

            llm_out = self._call_llm(system, user)
            hypothesis = llm_out.get("hypothesis", "")
            self.learned_insights = llm_out.get("learned_insights", self.learned_insights)

            genomes = []
            schema_failures = 0
            for raw in llm_out.get("genomes", []):
                g = repair_genome(raw)
                if g is None:
                    schema_failures += 1
                    continue
                genomes.append(g)

            # Elitism: carry the best 2 genomes forward untouched (no re-generation cost).
            candidates: list[CandidateResult] = list(self.elite)
            new_results = await self.generator.generate_batch(genomes)
            for genome, gen_result in zip(genomes, new_results):
                candidates.append(self._score_candidate(genome, gen_result.audio_path))

            scored = [c for c in candidates if c.D_brain is not None]
            best = min(scored, key=lambda c: c.D_brain) if scored else None
            mean_D = float(sum(c.D_brain for c in scored) / len(scored)) if scored else None

            gen_result = GenerationResult(
                generation_index=gen_idx, hypothesis=hypothesis, learned_insights=self.learned_insights,
                candidates=candidates, best=best, mean_D_brain=mean_D, elapsed_s=time.time() - t0,
            )
            self.history.append(gen_result)
            self._log_generation(gen_result)

            if scored:
                self.elite = sorted(scored, key=lambda c: c.D_brain)[:2]

            if self.on_generation:
                self.on_generation(gen_result)

            logger.info(
                "Generation %d: %d candidates, %d scored, %d schema failures, best D_brain=%s, hypothesis=%r",
                gen_idx, len(candidates), len(scored), schema_failures,
                f"{best.D_brain:.4f}" if best else None, hypothesis[:120],
            )

            stop_reason = self._should_stop()
            if stop_reason:
                logger.info("Stopping optimizer run: %s", stop_reason)
                break

        return self.history
