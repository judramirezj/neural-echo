"""Fast contract tests for the deployable Daniel optimization loop.

These checks intentionally avoid loading TRIBE or making paid API calls.
"""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from neural_echo import analysis, ingest
from neural_echo import optimizer as optimizer_module
from neural_echo.generator import Chunk, Genome, repair_genome
from neural_echo.optimizer import (
    MAX_GENERATION_ATTEMPTS,
    MODEL_ID,
    PATIENCE,
    OptimizerRun,
    parse_llm_json,
)


def _chunk(**overrides) -> dict:
    value = {
        "text": "[Intro]",
        "duration_ms": 30_000,
        "positive_styles": ["96 BPM", "warm analog polysynth"],
        "negative_styles": ["harsh clipping"],
        "context_adherence": "high",
    }
    value.update(overrides)
    return value


def test_daniel_runtime_constants_and_no_gemini_scoring_gate():
    assert MODEL_ID == "claude-sonnet-5"
    assert PATIENCE == 6
    assert MAX_GENERATION_ATTEMPTS == 6
    assert ingest.DEFAULT_WINDOW_S == 90.0
    assert not hasattr(analysis, "novelty_check")
    assert not hasattr(analysis, "constraint_adherence")


def test_music_v2_plan_contract_accepts_daniels_limits():
    styles = [f"technical cue {index}" for index in range(50)]
    plan = Genome(chunks=[Chunk(**_chunk(positive_styles=styles))])
    assert len(plan.chunks[0].positive_styles) == 50

    with pytest.raises(ValidationError):
        Genome(chunks=[Chunk(**_chunk(positive_styles=styles + ["one too many"]))])


def test_legacy_xhigh_is_repaired_but_not_emitted():
    repaired = repair_genome({"chunks": [_chunk(context_adherence="xhigh")]})
    assert repaired is not None
    assert repaired.chunks[0].context_adherence == "high"


def test_generation_failures_are_classified_like_daniels_loop():
    assert OptimizerRun._is_transient_generation_error("status_code: 500 internal_server_error")
    assert OptimizerRun._is_transient_generation_error("Connection timeout")
    assert OptimizerRun._is_content_policy_error("bad_composition_plan")
    assert OptimizerRun._is_content_policy_error("Terms of Service")


def test_json_parser_accepts_anthropic_blocks_and_wrappers():
    assert parse_llm_json('Here is the result: {"chunks": []} Thanks.') == {"chunks": []}
    assert parse_llm_json('```json\n{"change_seed": false}\n```') == {"change_seed": False}


def test_system_prompt_is_tradition_neutral_and_keeps_user_constraint():
    run = OptimizerRun.__new__(OptimizerRun)
    run.reference_analysis = {"duration_s": 90.0, "likely_has_vocals": False}
    run.constraint_text = "include the user's hand-drum pattern"

    prompt = run._system_prompt(iteration=1)
    normalized_prompt = " ".join(prompt.split())

    assert "ANY period or musical tradition" in prompt
    assert "rubato, free tempo" in prompt
    assert "Never default to contemporary pop" in prompt
    assert "include the user's hand-drum pattern" in prompt
    assert "specific musical change activates a specific brain region" in normalized_prompt


def test_one_iteration_graph_runs_end_to_end_without_paid_services(monkeypatch, tmp_path):
    class FakeModel:
        @staticmethod
        def get_events_dataframe(audio_path):
            return {"audio_path": audio_path}

        @staticmethod
        def predict(events):
            return np.zeros((4, 8), dtype=np.float32), None

    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        async def generate_one(self, genome, seed=None):
            return SimpleNamespace(error=None, audio_path="candidate.mp3")

    class FakeLlm:
        @staticmethod
        def invoke(messages):
            return AIMessage(content=json.dumps({"chunks": [_chunk()]}))

    fake_cost = SimpleNamespace(global_score=0.5)
    monkeypatch.setattr(optimizer_module.compat, "get_tribe_model", lambda: FakeModel())
    monkeypatch.setattr(optimizer_module.compat, "release_inference_memory", lambda: None)
    monkeypatch.setattr(optimizer_module.analysis, "analyze_reference", lambda _: {
        "duration_s": 30.0,
        "likely_has_vocals": False,
        "tempo_bpm": 96.0,
    })
    monkeypatch.setattr(optimizer_module.atlases, "build_lobule_regions", dict)
    monkeypatch.setattr(optimizer_module, "ElevenLabsGenerator", FakeGenerator)
    monkeypatch.setattr(optimizer_module.metric, "compute_cost", lambda *args: fake_cost)
    monkeypatch.setattr(optimizer_module.metric, "format_cost_for_llm", lambda *args, **kwargs: "raw cost")
    monkeypatch.setattr(
        optimizer_module,
        "summarize_vertex_residual",
        lambda *args: np.zeros(8, dtype=np.float32),
    )

    run = OptimizerRun(
        reference_audio_path="reference.wav",
        constraint_text="feature hand percussion",
        db_path=tmp_path / "run.sqlite3",
        dry_run=True,
        max_iterations=1,
        llm=FakeLlm(),
    )
    history = asyncio.run(run.run())

    assert len(history) == 1
    assert history[0].audio_path == "candidate.mp3"
    assert history[0].cost.global_score == 0.5
