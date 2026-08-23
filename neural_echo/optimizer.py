"""The closed-loop optimizer: a single-lineage LangGraph loop that mutates
one composition plan iteration by iteration — LLM proposes a plan ->
ElevenLabs renders it -> TRIBE scores it against the reference via
metric.compute_cost -> the LLM sees the full region x window diagnostic
matrix and writes a next plan. Runs until a success threshold, a patience
budget, or a max-iteration cap.

This mirrors daniel_algorithm.ipynb's pipeline_e2e.py algorithm: region-
parcellated raw cost, two-phase prompting, adaptive plan simplification on
ElevenLabs 5xx errors, retry/reformulate on content rejection, layered memory
compression, fixed-seed control, and patience-based stopping. The creative
constraint is prompt context only; every successfully rendered candidate is
scored directly, exactly as in Daniel's loop.

Every iteration's `reasoning`/`changes_summary` are first-class fields on
IterationResult specifically so callers (the API's SSE stream, the UI) can
surface "the reasoning log" live — this is what makes the optimizer legible
rather than a black box, and it is never dropped or summarized away between
here and the frontend.
"""
import asyncio
import json
import logging
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from . import analysis, atlases, compat, metric
from .brain_visualization import summarize_vertex_activity, summarize_vertex_residual
from .generator import ElevenLabsGenerator, Genome, repair_genome

logger = logging.getLogger(__name__)

MODEL_ID = "claude-sonnet-5"
MAX_ITERATIONS_DEFAULT = 10
SUCCESS_THRESHOLD = 0.15          # stop once best_score beats this (lower is better)
PATIENCE = 6                      # Daniel's updated loop allows six non-improving iterations
DETAIL_ITERATIONS = 3             # most recent iterations kept in full detail in `messages`
SUMMARIZE_TOKEN_THRESHOLD = 40_000
MAX_GENERATION_ATTEMPTS = 6       # shared budget for retry/reformulate/simplify recovery
PHASE_1_ITERATIONS = 3            # iterations 1..PHASE_1_ITERATIONS raise specificity; after that, optimize direction

CONTENT_POLICY_MARKERS = ("bad_composition_plan", "Terms of Service")

SYSTEM_PROMPT_TEMPLATE = """You are a professional audio director working with ElevenLabs Music v2.
You iteratively optimize a composition plan to minimize the global cost between a candidate
track and a brain-response benchmark (fMRI activity predicted by Meta's TRIBE v2 model from a
reference track), while satisfying the user's creative constraint.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THE INITIAL PLAN IS NOISY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The reference may come from ANY period or musical tradition: classical, folk and traditional
music from anywhere in the world, jazz, electronic, urban, rock, metal, experimental, or a
hybrid. Never default to contemporary pop, four-on-the-floor rhythm, synthesizers, or a standard
verse/chorus structure unless the reference evidence supports it.

A prompt answers five questions whether you address them or not: genre, mood, instrumentation,
tempo, and production era. Any dimension you leave open, ElevenLabs fills with "the most
statistically average choice" — which varies between generations and makes it impossible to
attribute score changes to your prompt vs. randomness.

Studio vocabulary moves real levers. The model respects exact numbers (BPM, Hz, dB, ms). Vague
adjectives (energetic, dark, powerful) have weak effect.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR STRATEGY, BY PHASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{current_phase_note}

PHASE 1 (iterations 1-3): RAISE SPECIFICITY
The initial plan is usually generic. Densify it with technical language native to the
REFERENCE'S musical tradition rather than forcing every track into the vocabulary of modern
electronic production:
  - Genre/tradition: an exact subgenre, ensemble, or compositional form
  - Mood: concrete descriptors appropriate to the source
  - Instrumentation: ensemble, instrument family, articulation, register, and timbre
  - Rhythm: exact BPM and meter when stable; otherwise state rubato, free tempo, swing,
    polyrhythm, or the relevant asymmetric meter
  - Era and space: period-appropriate recording character, acoustic space, or mixing approach
Examples across traditions include "baroque harpsichord sonata", "modal acoustic jazz quartet",
"West African polyrhythmic ensemble", "boom bap hip hop", "ambient drone", and "progressive
metal in 7/8". Use only the tradition actually supported by the reference.
Goal of this phase: reduce variance and anchor the generator to the correct musical world.

PHASE 2 (iteration 4+): OPTIMIZE DIRECTION
With the plan already specific, start moving parameters directed at the cost diagnostics.
Change 1-2 aspects per iteration so you can attribute the effect. If a change made the score
worse, revert and try another direction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELDS YOU CONTROL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per chunk:
  - text is what sounds literally: section labels, actual sung lyrics, phonetics like (oooh),
    and short inline cues like {{guitar solo}}. Never put style, production, mood, or
    instrumentation descriptions here or ElevenLabs may sing those words.
  - duration_ms: {min_chunk_ms}-{max_chunk_ms} (split chunks to introduce structure)
  - positive_styles: up to 50 terms. Prioritize TECHNICAL terms over adjectives.
  - negative_styles: up to 50 terms. Block cross-contamination.
  - context_adherence: "low" | "medium" | "high" (how strongly this chunk sticks to
    the context of the preceding ones)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO OPERATE PLAN COMPLEXITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The API permits up to 50 styles per chunk and 30 chunks, but ElevenLabs' practical complexity
ceiling changes with server load and parameter combinations.
  - If a plan caused repeated 5xx responses and was simplified, treat the simplified size as the
    current ceiling. Do not jump back up.
  - Increase detail gradually: at most 20-30% more styles or one extra chunk in one iteration.
  - Prefer more precise terms over more terms, especially during Phase 1.
  - During Phase 2, if error is falling, change content rather than increasing plan size.
  - Use the complexity statistics in each iteration result to decide whether to hold, raise, or
    lower complexity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESTRICTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hard (never violate):
  - Never use names of artists, composers, bands, songs, films, studios, venues, or commercial
    instrument brands. ElevenLabs may reject them. Replace proper names with precise generic
    descriptions such as "19th-century romantic symphony", "epic orchestral film score",
    "solo classical violin", "concert grand piano", or "analog drum machine". You may name
    genres, traditions, periods, and decades, but never people or brands.
  - The user's creative constraint below must be clearly satisfied:
    "{constraint_text}"
  - Reference vocal guidance: {vocal_guidance}

Soft (respect for coherence):
  - Stay inside the initial plan's musical world. Do not add distorted synthesizers to an
    acoustic jazz quartet or delicate harp to death metal unless that contrast is specifically
    supported by the original or the user's constraint. Refine within the tradition rather than
    restarting in a familiar default genre.
  - Total plan duration should stay close to the reference's duration ({target_duration_ms}ms).
    Don't compress a long arc into a short one — that destroys the temporal dynamics the
    benchmark captures.
  - You may (and often should) split the plan into several chunks to represent structure:
    intro, build, drop, sustain, breakdown, outro. Each chunk between {min_chunk_ms}-
    {max_chunk_ms}ms; total plan up to {max_total_ms}ms.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO REASON ABOUT THE BRAIN-COST MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - The matrix shows {n_windows} time windows x ~50 anatomical regions. Each cell: candidate /
    target / difference. Lower per-region score is better (it's distance + (1 - arc
    correlation) between the candidate's and the reference's temporal arc in that region).
  - You may use general neuroscience intuition ONLY to prioritize what to try first (e.g.
    auditory cortices process timbre; motor areas respond to rhythm).
  - The only source of truth is next iteration's matrix. Never claim that a specific musical
    change activates a specific brain region — there is no evidence supporting that level of
    causality. This applies equally to acoustic, orchestral, vocal, electronic, and traditional
    instruments.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT (mandatory, ONLY valid JSON, no extra text)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "reasoning": "1-2 sentences: what you observed, what phase you're in, what you're trying",
  "changes_summary": "brief description of what you changed vs. the previous plan",
  "plan": {{ ...full composition plan (chunks)... }},
  "change_seed": false
}}

change_seed:
  - false by default (keeps the seed fixed for clean attribution)
  - true ONLY if you've had {patience}+ iterations without improvement and already tried
    significant prompt changes. Explain why in "reasoning"."""


def response_text(content) -> str:
    """ChatAnthropic's AIMessage.content is a plain str for simple replies but a
    list of content blocks (e.g. [{"type": "text", "text": "..."}]) whenever the
    underlying API response has multiple blocks — normalize both shapes to a
    single string everywhere a response is parsed as text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def parse_llm_json(text: str) -> dict:
    """Extract one JSON object from Claude's plain-text response.

    Sonnet can occasionally surround valid JSON with prose or a Markdown
    fence. The optimizer retries malformed JSON, but accepting these harmless
    wrappers avoids wasting an LLM call and matches the final notebook logic.
    """
    text = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end >= start:
            text = text[start:end + 1]
    return json.loads(text)


@dataclass
class IterationResult:
    iteration_index: int
    reasoning: str
    changes_summary: str
    plan: Genome
    seed: int
    audio_path: str | None
    is_best: bool
    elapsed_s: float
    cost: metric.CostResult | None = None
    # Internal compact visualization payload; intentionally omitted from SSE.
    brain_residual: np.ndarray | None = None
    brain_reference_activity: np.ndarray | None = None
    brain_candidate_activity: np.ndarray | None = None


class OptimizerState(TypedDict):
    plan: dict                       # {"chunks": [...]}, JSON-serializable Genome
    seed: int
    iteration: int
    benchmark_preds: np.ndarray
    best_score: float
    best_iteration: int | None
    best_plan: dict | None
    best_audio_path: str | None
    iterations_without_improvement: int
    messages: list[BaseMessage]
    memory_summary: str


class OptimizerRun:
    """One end-to-end optimization run against a single reference clip."""

    def __init__(
        self,
        reference_audio_path: str,
        constraint_text: str,
        db_path: Path,
        dry_run: bool = False,
        stub_clips_dir: Path | None = None,
        max_iterations: int = MAX_ITERATIONS_DEFAULT,
        llm: ChatAnthropic | None = None,
        on_iteration=None,  # optional callback(IterationResult) for live SSE streaming
    ):
        self.reference_audio_path = reference_audio_path
        self.constraint_text = constraint_text
        self.max_iterations = max_iterations
        self.on_iteration = on_iteration

        self.model = compat.get_tribe_model()
        self.llm = llm or ChatAnthropic(model=MODEL_ID, max_tokens=4000)
        self.generator = ElevenLabsGenerator(
            output_dir=Path("data/generated"), dry_run=dry_run, stub_clips_dir=stub_clips_dir,
        )
        self.regions = atlases.build_lobule_regions()

        self.db_path = db_path
        self._init_db()

        self.reference_analysis = analysis.analyze_reference(reference_audio_path)

        self.history: list[IterationResult] = []
        # Set by _node_propose_next_plan for the iteration it just produced a plan
        # for; consumed by the following _node_generate_and_score call.
        self._pending_reasoning = ""
        self._pending_changes_summary = "Initial plan built by Claude from reference audio analysis."
        self.graph = self._build_graph()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration_index INTEGER,
                reasoning TEXT,
                changes_summary TEXT,
                plan_json TEXT,
                seed INTEGER,
                audio_path TEXT,
                is_best INTEGER,
                elapsed_s REAL,
                global_score REAL,
                created_at REAL
            )
        """)
        conn.commit()
        conn.close()

    def _log_iteration(self, r: IterationResult):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO iterations
            (iteration_index, reasoning, changes_summary, plan_json, seed, audio_path, is_best,
             elapsed_s, global_score, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                r.iteration_index, r.reasoning, r.changes_summary, r.plan.model_dump_json(), r.seed,
                r.audio_path, int(r.is_best), r.elapsed_s, r.cost.global_score if r.cost else None,
                time.time(),
            ),
        )
        conn.commit()
        conn.close()

    def _llm_json(self, context: list[BaseMessage], max_attempts: int = 3) -> tuple[str, dict]:
        """Invoke the LLM and parse its response as JSON, retrying with a
        corrective follow-up message if the response isn't valid JSON. Claude
        occasionally garbles strict JSON on a long, complex plan despite
        instructions — a single-lineage run has no other candidate to fall back
        on, so this failure mode has to be handled here rather than left to
        crash the whole run. Returns (raw_response_text, parsed_dict)."""
        messages = list(context)
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            response = self.llm.invoke(messages)
            text = response_text(response.content)
            try:
                return text, parse_llm_json(text)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("LLM response wasn't valid JSON (attempt %d/%d): %s", attempt + 1, max_attempts, e)
                messages = messages + [
                    AIMessage(content=text),
                    HumanMessage(content=(
                        f"Your last response was not valid JSON ({e}). Respond again with ONLY the "
                        "valid JSON object, no extra text, no markdown fences."
                    )),
                ]
        raise RuntimeError(f"LLM did not return valid JSON after {max_attempts} attempts: {last_error}")

    def _system_prompt(self, iteration: int) -> str:
        from .generator import MAX_CHUNK_S, MAX_TOTAL_S, MIN_CHUNK_S

        target_duration_ms = int(self.reference_analysis["duration_s"] * 1000)
        current_phase_note = (
            f"You are currently on iteration {iteration}, which is in PHASE 1 (raise specificity)."
            if iteration <= PHASE_1_ITERATIONS else
            f"You are currently on iteration {iteration}, which is in PHASE 2 (optimize direction)."
        )
        return SYSTEM_PROMPT_TEMPLATE.format(
            current_phase_note=current_phase_note,
            constraint_text=self.constraint_text,
            vocal_guidance=self._vocal_guidance(),
            target_duration_ms=target_duration_ms,
            min_chunk_ms=MIN_CHUNK_S * 1000,
            max_chunk_ms=MAX_CHUNK_S * 1000,
            max_total_ms=MAX_TOTAL_S * 1000,
            n_windows=metric.N_TIME_WINDOWS,
            patience=PATIENCE,
        )

    def _vocal_guidance(self) -> str:
        if self.reference_analysis["likely_has_vocals"]:
            return (
                "The reference likely contains vocals. Put actual singable lyrics, phonetics, or "
                "spoken lines in chunk text as appropriate; describe how the voice sounds only in "
                "positive_styles. Claude should infer a fitting treatment from the measured features "
                "and the user's creative constraint."
            )
        return (
            "The reference appears instrumental. Keep chunk text to section labels and brief inline "
            "cues; do not add lyrics."
        )

    def _build_initial_plan(self) -> Genome:
        target_duration_ms = int(self.reference_analysis["duration_s"] * 1000)
        system = (
            "You are an audio director. Build an ElevenLabs Music v2 composition plan aiming to "
            "generate a track in the same sonic world as a reference track, described below by "
            "measured audio features. Also satisfy the user's "
            "creative constraint.\n\n"
            "The reference can belong to ANY period or musical tradition: classical, folk/world, "
            "jazz, electronic, urban, rock, metal, experimental, or hybrid. Do not default to pop, "
            "electronic instrumentation, 4/4, or verse/chorus form. Use only what the measurements "
            "and user context support; if tempo is unstable, prefer rubato/free-tempo language over "
            "inventing a rigid BPM.\n\n"
            "Rules:\n"
            "- Never use names of artists, composers, bands, songs, films, studios, venues, or "
            "commercial instrument brands. Describe the musical property generically instead.\n"
            f"- Vocal handling: {self._vocal_guidance()}\n"
            "- Describe voices by audible register, delivery, articulation, language, and production; "
            "do not invent performer identity or demographic attributes.\n"
            "- Chunk text is literal output: section labels, actual lyrics/phonetics, and short inline "
            "cues only. Put genre, mood, instrumentation, and production in styles, never in text.\n"
            "- Each chunk: 3000-120000ms. Total plan duration should approximate the target below.\n"
            "- context_adherence: \"low\" | \"medium\" | \"high\".\n"
            "- Start conservatively with 2-4 chunks, 10-15 positive styles and 5-8 negative styles "
            "per chunk, though the API maximum is 50 each.\n"
            "- Use tradition-appropriate technical vocabulary: ensemble and articulation, stable BPM "
            "or rubato/free tempo, meter or rhythmic feel, acoustic space, period, and production.\n\n"
            "Respond with ONLY the JSON plan, no extra text, in this shape:\n"
            '{"chunks": [{"text": "[Genre-appropriate section]", "duration_ms": <int>, '
            '"positive_styles": ["..."], "negative_styles": ["..."], "context_adherence": "high"}, ...]}'
        )
        user = json.dumps({
            "reference_track_analysis": self.reference_analysis,
            "user_constraint": self.constraint_text,
            "target_duration_ms": target_duration_ms,
        }, indent=2)
        _, parsed = self._llm_json([SystemMessage(content=system), HumanMessage(content=user)])
        genome = repair_genome(parsed)
        if genome is None:
            raise RuntimeError(f"Initial plan failed validation and could not be repaired: {parsed}")
        return genome

    def _request_reformulation(self, messages: list[BaseMessage], plan: Genome, problem: str, iteration: int) -> Genome | None:
        """Ask Claude to revise the same plan after an ElevenLabs rejection."""
        context = [
            SystemMessage(content=self._system_prompt(iteration=iteration)),
            *messages,
            HumanMessage(content=(
                f"The plan you proposed was rejected:\n{plan.model_dump_json(indent=2)}\n\n"
                f"Problem: {problem}\n\n"
                "Revise the SAME plan (same direction, same goals) to fix this specific problem. "
                "Respond with ONLY the usual JSON."
            )),
        ]
        try:
            _, parsed = self._llm_json(context)
            revised = repair_genome(parsed.get("plan", parsed))
            if revised is None:
                logger.warning("Reformulation response failed validation/repair: %s", parsed)
            return revised
        except RuntimeError as e:
            logger.warning("Could not get a valid reformulation: %s", e)
            return None

    def _request_simplification(self, plan: Genome, iteration: int) -> Genome | None:
        n_chunks = len(plan.chunks)
        n_styles = sum(len(chunk.positive_styles) for chunk in plan.chunks)
        context = [
            SystemMessage(content=self._system_prompt(iteration=iteration)),
            HumanMessage(content=(
                f"This plan repeatedly caused an ElevenLabs 5xx/internal-server error:\n"
                f"{plan.model_dump_json(indent=2)}\n\n"
                f"It currently has {n_chunks} chunks and {n_styles} positive styles total. This "
                "usually means the plan is too complex for the backend's current ceiling. Return a "
                "SIMPLER version in the same sonic direction: reduce chunks when there are many, "
                "reduce styles per chunk, and shorten the total duration if it exceeds three minutes. "
                "Keep the most specific terms and remove redundancies. Respond ONLY with the usual JSON."
            )),
        ]
        try:
            _, parsed = self._llm_json(context)
            simplified = repair_genome(parsed.get("plan", parsed))
            if simplified is not None:
                logger.info(
                    "Simplified failed plan from %d chunks/%d styles to %d chunks/%d styles",
                    n_chunks, n_styles, len(simplified.chunks),
                    sum(len(chunk.positive_styles) for chunk in simplified.chunks),
                )
            return simplified
        except RuntimeError as e:
            logger.warning("Could not obtain a valid simplified plan: %s", e)
            return None

    @staticmethod
    def _is_content_policy_error(message: str) -> bool:
        return any(marker in message for marker in CONTENT_POLICY_MARKERS)

    @staticmethod
    def _is_transient_generation_error(message: str) -> bool:
        lowered = message.lower()
        return (
            "status_code: 5" in message
            or "internal_server_error" in lowered
            or "timeout" in lowered
            or "connection" in lowered
        )

    def _cost_message(self, plan: Genome, cost: metric.CostResult, iteration: int) -> str:
        n_positive = sum(len(chunk.positive_styles) for chunk in plan.chunks)
        n_negative = sum(len(chunk.negative_styles) for chunk in plan.chunks)
        duration_ms = sum(chunk.duration_ms for chunk in plan.chunks)
        return (
            f"Iteration {iteration} completed.\n"
            f"COMPLEXITY of the plan that executed successfully: {len(plan.chunks)} chunks, "
            f"{n_positive} positive styles, {n_negative} negative styles, "
            f"total duration {duration_ms}ms.\n"
            f"Plan used:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Result:\n{metric.format_cost_for_llm(cost, iteration=iteration)}"
        )

    async def _node_generate_and_score(self, state: OptimizerState) -> dict:
        n = state["iteration"]
        genome = Genome.model_validate(state["plan"])
        seed = state["seed"]
        messages = state["messages"]
        t0 = time.time()
        reasoning, changes_summary = self._pending_reasoning, self._pending_changes_summary

        logger.info("Iteration %d/%d starting (seed=%d)", n, self.max_iterations, seed)

        audio_path = None
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            gen_result = await self.generator.generate_one(genome, seed=seed)
            if not gen_result.error:
                audio_path = gen_result.audio_path
                break
            if attempt == MAX_GENERATION_ATTEMPTS:
                raise RuntimeError(
                    f"Iteration {n}: ElevenLabs failed after {MAX_GENERATION_ATTEMPTS} attempts: "
                    f"{gen_result.error}"
                )
            if self._is_transient_generation_error(gen_result.error):
                if attempt == 1:
                    logger.warning(
                        "Iteration %d: transient ElevenLabs failure; retrying same plan in 10s", n
                    )
                    await asyncio.sleep(10)
                    continue
                simplified = self._request_simplification(genome, iteration=n)
                if simplified is None:
                    raise RuntimeError(
                        f"Iteration {n}: Claude could not simplify a plan after ElevenLabs 5xx"
                    )
                genome = simplified
                continue
            if self._is_content_policy_error(gen_result.error):
                logger.info("Iteration %d: ToS/copyright rejection, asking LLM to reformulate (attempt %d)", n, attempt)
                revised = self._request_reformulation(
                    messages, genome,
                    f"ElevenLabs rejected this plan for a Terms of Service violation ('{gen_result.error}'). "
                    "This usually means the plan included a restricted proper noun (an artist, composer, "
                    "song, film, studio, venue, or commercial instrument brand). Replace every proper noun "
                    "with an equivalent precise generic description.",
                    iteration=n,
                )
                if revised is None:
                    raise RuntimeError(
                        f"Iteration {n}: Claude could not reformulate a rejected ElevenLabs plan"
                    )
                genome = revised
                continue
            raise RuntimeError(f"Iteration {n}: ElevenLabs generation failed: {gen_result.error}")

        if audio_path is None:
            raise RuntimeError(f"Iteration {n}: ElevenLabs produced no audio")

        # Daniel's loop scores every successfully generated candidate directly.
        df = None
        auxiliary_output = None
        try:
            df = self.model.get_events_dataframe(audio_path=audio_path)
            preds, auxiliary_output = self.model.predict(events=df)
            preds = np.asarray(preds)
        finally:
            del df, auxiliary_output
            compat.release_inference_memory()
        cost = metric.compute_cost(preds, state["benchmark_preds"], self.regions)

        is_best = cost.global_score < state["best_score"]
        result = IterationResult(
            iteration_index=n, reasoning=reasoning, changes_summary=changes_summary, plan=genome, seed=seed,
            audio_path=audio_path, is_best=is_best, elapsed_s=time.time() - t0, cost=cost,
            brain_residual=summarize_vertex_residual(preds, state["benchmark_preds"]),
            brain_reference_activity=summarize_vertex_activity(state["benchmark_preds"]),
            brain_candidate_activity=summarize_vertex_activity(preds),
        )
        self._finish_iteration(result)

        updates = {
            "messages": messages + [HumanMessage(content=self._cost_message(genome, cost, n))],
            "iterations_without_improvement": 0 if is_best else state["iterations_without_improvement"] + 1,
        }
        if is_best:
            updates.update(best_score=cost.global_score, best_iteration=n,
                            best_plan=genome.model_dump(), best_audio_path=audio_path)
        return updates

    def _finish_iteration(self, result: IterationResult):
        self.history.append(result)
        self._log_iteration(result)
        if self.on_iteration:
            self.on_iteration(result)
        logger.info(
            "Iteration %d done: cost=%s is_best=%s",
            result.iteration_index,
            f"{result.cost.global_score:.4f}" if result.cost else None, result.is_best,
        )

    def _node_compress_memory(self, state: OptimizerState) -> dict:
        messages = state["messages"]
        approx_tokens = sum(len(m.content) for m in messages) // 4
        keep_count = DETAIL_ITERATIONS * 2
        if approx_tokens < SUMMARIZE_TOKEN_THRESHOLD or len(messages) <= keep_count:
            return {}

        to_summarize, to_keep = messages[:-keep_count], messages[-keep_count:]
        logger.info("Compressing %d old messages (~%d tokens estimated)", len(to_summarize), approx_tokens)

        summary_request = HumanMessage(content=(
            "Summarize the iterations above as a compact TABLE. One row per iteration, columns: "
            "iteration, changes_made, resulting_score, effect (improved/worsened/same). Max 15 lines "
            "total. This is memory — the LLM will use it to avoid repeating failed attempts."
        ))
        response = self.llm.invoke(
            [SystemMessage(content="Summarizer of experimental history.")] + to_summarize + [summary_request]
        )
        new_summary = (
            (state["memory_summary"] + "\n\n" if state["memory_summary"] else "")
            + f"[Summary through iteration {state['iteration'] - len(to_keep) // 2}]\n{response_text(response.content)}"
        )
        return {"messages": to_keep, "memory_summary": new_summary}

    def _node_propose_next_plan(self, state: OptimizerState) -> dict:
        # The system prompt's phase note is for the iteration this plan will run
        # in (state["iteration"] + 1), not the one that just finished.
        context = [SystemMessage(content=self._system_prompt(state["iteration"] + 1))]
        if state["memory_summary"]:
            context.append(HumanMessage(content=f"SUMMARY OF EARLIER ITERATIONS (compressed):\n{state['memory_summary']}"))
        context.extend(state["messages"])
        context.append(HumanMessage(content=(
            f"Iteration {state['iteration'] + 1}/{self.max_iterations}. "
            f"Best score so far: {state['best_score']:.4f} (lower is better). "
            "Propose the next plan. ONLY JSON."
        )))

        response_str, parsed = self._llm_json(context)
        genome = repair_genome(parsed.get("plan", {}))
        if genome is None:
            raise RuntimeError(f"Proposed plan failed validation and could not be repaired: {parsed}")

        reasoning = parsed.get("reasoning", "")
        changes_summary = parsed.get("changes", parsed.get("changes_summary", ""))
        logger.info("Iteration %d proposal — reasoning: %s", state["iteration"] + 1, reasoning[:160])

        new_seed = state["seed"]
        if parsed.get("change_seed"):
            new_seed = random.randint(1, 2**31 - 1)
            logger.info("LLM requested a seed change -> %d", new_seed)

        # Stash reasoning/changes_summary for the NEXT generate_and_score call to
        # attach to its IterationResult (the LLM's rationale for the plan it just
        # produced belongs to the iteration that plan runs in, not this one).
        self._pending_reasoning = reasoning
        self._pending_changes_summary = changes_summary

        return {
            "plan": genome.model_dump(),
            "iteration": state["iteration"] + 1,
            "seed": new_seed,
            "messages": state["messages"] + [AIMessage(content=response_str)],
        }

    def _should_continue(self, state: OptimizerState) -> str:
        if state["best_score"] < SUCCESS_THRESHOLD:
            logger.info("Success threshold reached (%.4f < %.4f)", state["best_score"], SUCCESS_THRESHOLD)
            return "stop"
        if state["iterations_without_improvement"] >= PATIENCE:
            logger.info("Patience exhausted (%d iterations without improvement)", state["iterations_without_improvement"])
            return "stop"
        if state["iteration"] >= self.max_iterations:
            logger.info("Max iterations reached (%d)", self.max_iterations)
            return "stop"
        return "continue"

    def _build_graph(self):
        graph = StateGraph(OptimizerState)
        graph.add_node("generate_and_score", self._node_generate_and_score)
        graph.add_node("compress_memory", self._node_compress_memory)
        graph.add_node("propose_next_plan", self._node_propose_next_plan)
        graph.set_entry_point("generate_and_score")
        graph.add_edge("generate_and_score", "compress_memory")
        graph.add_conditional_edges(
            "compress_memory", self._should_continue,
            {"continue": "propose_next_plan", "stop": END},
        )
        graph.add_edge("propose_next_plan", "generate_and_score")
        return graph.compile()

    async def run(self) -> list[IterationResult]:
        logger.info("Building initial plan with Claude from reference audio analysis...")
        initial_genome = self._build_initial_plan()

        logger.info("Running TRIBE on the reference (benchmark)...")
        df_ref = None
        auxiliary_output = None
        try:
            df_ref = self.model.get_events_dataframe(audio_path=self.reference_audio_path)
            bench_preds, auxiliary_output = self.model.predict(events=df_ref)
            benchmark_preds = np.asarray(bench_preds)
        finally:
            del df_ref, auxiliary_output
            compat.release_inference_memory()

        initial_state: OptimizerState = {
            "plan": initial_genome.model_dump(),
            "seed": random.randint(1, 2**31 - 1),
            "iteration": 1,
            "benchmark_preds": benchmark_preds,
            "best_score": float("inf"),
            "best_iteration": None,
            "best_plan": None,
            "best_audio_path": None,
            "iterations_without_improvement": 0,
            "messages": [],
            "memory_summary": "",
        }

        # recursion_limit: 3 nodes per iteration, generous headroom over max_iterations.
        await self.graph.ainvoke(initial_state, config={"recursion_limit": self.max_iterations * 3 + 10})
        return self.history
