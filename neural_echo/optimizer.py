"""The closed-loop optimizer: a single-lineage LangGraph loop that mutates
one composition plan iteration by iteration — LLM proposes a plan ->
ElevenLabs renders it -> TRIBE scores it against the reference via
metric.compute_cost -> the LLM sees the full region x window diagnostic
matrix and writes a next plan. Runs until a success threshold, a patience
budget, or a max-iteration cap.

This mirrors daniel_algorithm.ipynb's pipeline_e2e.py algorithm (region-
parcellated, time-windowed cost; two-phase prompting; retry/reformulate on
ElevenLabs rejection; layered memory compression; seed control) with two
additions the notebook doesn't have: the user's creative constraint and a
novelty ("not a near-cover of the reference") hard filter, both reused from
analysis.py and enforced the same way the notebook already handles ToS
rejections — ask the LLM to reformulate, then retry, rather than blending
them into the score.

Every iteration's `reasoning`/`changes_summary` are first-class fields on
IterationResult specifically so callers (the API's SSE stream, the UI) can
surface "the reasoning log" live — this is what makes the optimizer legible
rather than a black box, and it is never dropped or summarized away between
here and the frontend.
"""
import json
import logging
import random
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
from .brain_visualization import summarize_vertex_residual
from .generator import ElevenLabsGenerator, Genome, repair_genome

logger = logging.getLogger(__name__)

MODEL_ID = "claude-sonnet-5"
MAX_ITERATIONS_DEFAULT = 10
SUCCESS_THRESHOLD = 0.15          # stop once best_score beats this (lower is better)
PATIENCE = 3                      # stop after this many iterations without improvement
DETAIL_ITERATIONS = 3             # most recent iterations kept in full detail in `messages`
SUMMARIZE_TOKEN_THRESHOLD = 40_000
MAX_REFORMULATION_ATTEMPTS = 3    # bounded retries when ElevenLabs rejects a plan outright
PHASE_1_ITERATIONS = 3            # iterations 1..PHASE_1_ITERATIONS raise specificity; after that, optimize direction

CONTENT_POLICY_MARKERS = ("bad_composition_plan", "Terms of Service")

SYSTEM_PROMPT_TEMPLATE = """You are a professional audio director working with ElevenLabs Music v2.
You iteratively optimize a composition plan to minimize the global cost between a candidate
track and a brain-response benchmark (fMRI activity predicted by Meta's TRIBE v2 model from a
reference track), while satisfying the user's creative constraint.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THE INITIAL PLAN IS NOISY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
The initial plan is generic. Before changing direction, densify the existing plan by covering
all five dimensions with concrete technical vocabulary:
  - Genre: specific subgenre, not a broad label
  - Mood: concrete, compound descriptors
  - Instrumentation: specific generic instrument/synth type (never a brand or model name)
  - Tempo: exact BPM
  - Production era: decade + mixing school
You can also add production terminology: sidechain compression, parallel bus, plate reverb,
tape saturation, mid/side EQ, transient shaping, stereo width, headroom, integrated LUFS.
Goal of this phase: reduce variance, not change direction yet.

PHASE 2 (iteration 4+): OPTIMIZE DIRECTION
With the plan already specific, start moving parameters directed at the cost diagnostics.
Change 1-2 aspects per iteration so you can attribute the effect. If a change made the score
worse, revert and try another direction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELDS YOU CONTROL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per chunk:
  - text: section description; you can use inline {{braces}} for timing markers like
    "{{0:15 sub bass enters}}"
  - duration_ms: {min_chunk_ms}-{max_chunk_ms} (split chunks to introduce structure)
  - positive_styles: up to 12 terms. Prioritize TECHNICAL terms over adjectives.
  - negative_styles: up to 12 terms. Block cross-contamination.
  - context_adherence: "low" | "medium" | "high" | "xhigh" (how strongly this chunk sticks to
    the context of the preceding ones)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESTRICTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hard (never violate):
  - Never use proper nouns with copyright: artists, bands, songs, films, or branded
    instrument/synth/drum-machine names. ElevenLabs automatically rejects a plan that contains
    them. Example: NOT "Juno-60 pad" -> "warm analog polysynth pad"; NOT "TR-909 kick" ->
    "punchy analog drum machine kick". You may name genres and eras/decades, never brands or
    people.
  - The user's creative constraint below must be clearly satisfied:
    "{constraint_text}"

Soft (respect for coherence):
  - Stay within the same creative direction across iterations unless a hypothesis calls for a
    deliberate change — you're refining, not restarting from scratch every iteration.
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
  - The only source of truth is next iteration's matrix. Never invent precise causal claims
    ("raising the hi-hat activates the right parietal lobe") — there's no literature backing
    that specificity.

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
    """Best-effort JSON parse tolerating ```json ... ``` fences the LLM sometimes wraps
    its response in."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json")
        text = text.strip()
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
    rejected_reason: str | None = None  # "generation_failed" | "near_cover" | "constraint_not_met"
    adherence: float | None = None
    novelty_audio_sim: float | None = None
    is_near_cover: bool | None = None
    # Internal compact visualization payload; intentionally omitted from SSE.
    brain_residual: np.ndarray | None = None


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
        adherence_tau: float = 0.15,
        llm: ChatAnthropic | None = None,
        on_iteration=None,  # optional callback(IterationResult) for live SSE streaming
    ):
        self.reference_audio_path = reference_audio_path
        self.constraint_text = constraint_text
        self.max_iterations = max_iterations
        self.adherence_tau = adherence_tau
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
        self.reference_clap_embedding = analysis.clap_audio_embedding(reference_audio_path)

        self.history: list[IterationResult] = []
        # Set by _node_propose_next_plan for the iteration it just produced a plan
        # for; consumed by the following _node_generate_and_score call.
        self._pending_reasoning = ""
        self._pending_changes_summary = "Initial plan built from reference audio analysis."
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
                rejected_reason TEXT,
                adherence REAL,
                novelty_audio_sim REAL,
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
             elapsed_s, global_score, rejected_reason, adherence, novelty_audio_sim, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r.iteration_index, r.reasoning, r.changes_summary, r.plan.model_dump_json(), r.seed,
                r.audio_path, int(r.is_best), r.elapsed_s, r.cost.global_score if r.cost else None,
                r.rejected_reason, r.adherence, r.novelty_audio_sim, time.time(),
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
            target_duration_ms=target_duration_ms,
            min_chunk_ms=MIN_CHUNK_S * 1000,
            max_chunk_ms=MAX_CHUNK_S * 1000,
            max_total_ms=MAX_TOTAL_S * 1000,
            n_windows=metric.N_TIME_WINDOWS,
            patience=PATIENCE,
        )

    def _build_initial_plan(self) -> Genome:
        target_duration_ms = int(self.reference_analysis["duration_s"] * 1000)
        system = (
            "You are an audio director. Build an ElevenLabs Music v2 composition plan aiming to "
            "generate a track in the same sonic world as a reference track, described below by its "
            "extracted audio features (not by listening to it directly), while satisfying the user's "
            "creative constraint.\n\n"
            "Rules:\n"
            "- Never use proper nouns with copyright (artists, bands, songs, films, branded "
            "instrument/synth/drum-machine names) — ElevenLabs rejects plans that contain them.\n"
            "- Each chunk: 3000-120000ms. Total plan duration should approximate the target below.\n"
            "- context_adherence: \"low\" | \"medium\" | \"high\" | \"xhigh\".\n"
            "- Use concrete technical vocabulary (exact BPM, production techniques, specific generic "
            "instrument types) — not vague adjectives.\n\n"
            "Respond with ONLY the JSON plan, no extra text, in this shape:\n"
            '{"chunks": [{"text": "[Intro/Build/Drop/Outro/...]", "duration_ms": <int>, '
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
        """Shared reformulation path for ElevenLabs ToS rejections, near-cover
        candidates, and constraint-adherence failures: ask the LLM to revise the
        SAME plan to fix the specific problem, then validate/repair the response.
        Returns None if the LLM's revision can't be salvaged."""
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

    @staticmethod
    def _is_content_policy_error(message: str) -> bool:
        return any(marker in message for marker in CONTENT_POLICY_MARKERS)

    def _cost_message(self, plan: Genome, cost: metric.CostResult, iteration: int) -> str:
        return (
            f"Iteration {iteration} completed.\n"
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
        for attempt in range(1, MAX_REFORMULATION_ATTEMPTS + 1):
            gen_result = await self.generator.generate_one(genome, seed=seed)
            if not gen_result.error:
                audio_path = gen_result.audio_path
                break
            if self._is_content_policy_error(gen_result.error) and attempt < MAX_REFORMULATION_ATTEMPTS:
                logger.info("Iteration %d: ToS/copyright rejection, asking LLM to reformulate (attempt %d)", n, attempt)
                revised = self._request_reformulation(
                    messages, genome,
                    f"ElevenLabs rejected this plan for a Terms of Service violation ('{gen_result.error}'). "
                    "This almost always means the plan included a proper noun with copyright (an artist, "
                    "song, film, or branded instrument/synth/drum-machine name). Replace any proper nouns "
                    "with equivalent generic descriptions.",
                    iteration=n,
                )
                if revised is None:
                    break
                genome = revised
                continue
            raise RuntimeError(f"Iteration {n}: ElevenLabs generation failed: {gen_result.error}")

        if audio_path is None:
            return self._rejection_update(state, genome, seed, "generation_failed", t0)

        # Hard filters (never blended into the score) — reused from analysis.py.
        novelty = analysis.novelty_check(
            audio_path, self.reference_audio_path, self.reference_analysis,
            reference_embedding=self.reference_clap_embedding,
        )
        if novelty["is_near_cover"]:
            logger.info("Iteration %d: near-cover (sim=%.2f), asking LLM to reformulate", n, novelty["audio_similarity"])
            revised = self._request_reformulation(
                messages, genome,
                f"The generated candidate was judged a near-cover of the reference (audio similarity "
                f"{novelty['audio_similarity']:.2f}, tempo delta {novelty['tempo_delta_frac']:.1%}). "
                "Revise the plan to be more clearly original while keeping the same creative direction.",
                iteration=n,
            )
            if revised is not None:
                retry_result = await self.generator.generate_one(revised, seed=seed)
                if not retry_result.error:
                    genome, audio_path = revised, retry_result.audio_path
                    novelty = analysis.novelty_check(
                        audio_path, self.reference_audio_path, self.reference_analysis,
                        reference_embedding=self.reference_clap_embedding,
                    )
            if novelty["is_near_cover"]:
                return self._rejection_update(state, genome, seed, "near_cover", t0, audio_path=audio_path, novelty=novelty)

        adherence = analysis.constraint_adherence(audio_path, self.constraint_text)
        if adherence < self.adherence_tau:
            logger.info("Iteration %d: constraint not met (adherence=%.2f), asking LLM to reformulate", n, adherence)
            revised = self._request_reformulation(
                messages, genome,
                f"The generated candidate scored {adherence:.2f} on adherence to the user's creative "
                f"constraint ('{self.constraint_text}'), below the required threshold "
                f"{self.adherence_tau:.2f}. Revise the plan to satisfy this constraint more strongly.",
                iteration=n,
            )
            if revised is not None:
                retry_result = await self.generator.generate_one(revised, seed=seed)
                if not retry_result.error:
                    genome, audio_path = revised, retry_result.audio_path
                    adherence = analysis.constraint_adherence(audio_path, self.constraint_text)
            if adherence < self.adherence_tau:
                return self._rejection_update(state, genome, seed, "constraint_not_met", t0, audio_path=audio_path, adherence=adherence, novelty=novelty)

        # TRIBE + brain cost.
        df = self.model.get_events_dataframe(audio_path=audio_path)
        preds, _ = self.model.predict(events=df)
        preds = np.asarray(preds)
        cost = metric.compute_cost(preds, state["benchmark_preds"], self.regions)

        is_best = cost.global_score < state["best_score"]
        result = IterationResult(
            iteration_index=n, reasoning=reasoning, changes_summary=changes_summary, plan=genome, seed=seed,
            audio_path=audio_path, is_best=is_best, elapsed_s=time.time() - t0, cost=cost,
            adherence=adherence, novelty_audio_sim=novelty["audio_similarity"], is_near_cover=False,
            brain_residual=summarize_vertex_residual(preds, state["benchmark_preds"]),
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

    def _rejection_update(
        self, state: OptimizerState, genome: Genome, seed: int, reason: str, t0: float,
        audio_path: str | None = None, adherence: float | None = None, novelty: dict | None = None,
    ) -> dict:
        n = state["iteration"]
        result = IterationResult(
            iteration_index=n, reasoning=self._pending_reasoning, changes_summary=self._pending_changes_summary,
            plan=genome, seed=seed, audio_path=audio_path, is_best=False, elapsed_s=time.time() - t0,
            rejected_reason=reason, adherence=adherence,
            novelty_audio_sim=(novelty or {}).get("audio_similarity"),
            is_near_cover=(novelty or {}).get("is_near_cover"),
        )
        self._finish_iteration(result)
        summary = (
            f"Iteration {n} rejected ({reason}).\nPlan used:\n{genome.model_dump_json(indent=2)}\n\n"
            "This candidate was never scored against the brain-response benchmark — fix the "
            "underlying problem in your next plan."
        )
        return {
            "messages": state["messages"] + [HumanMessage(content=summary)],
            "iterations_without_improvement": state["iterations_without_improvement"] + 1,
        }

    def _finish_iteration(self, result: IterationResult):
        self.history.append(result)
        self._log_iteration(result)
        if self.on_iteration:
            self.on_iteration(result)
        logger.info(
            "Iteration %d done: rejected=%s cost=%s is_best=%s",
            result.iteration_index, result.rejected_reason,
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
        logger.info("Building initial plan from reference audio analysis...")
        initial_genome = self._build_initial_plan()

        logger.info("Running TRIBE on the reference (benchmark)...")
        df_ref = self.model.get_events_dataframe(audio_path=self.reference_audio_path)
        bench_preds, _ = self.model.predict(events=df_ref)
        benchmark_preds = np.asarray(bench_preds)

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
