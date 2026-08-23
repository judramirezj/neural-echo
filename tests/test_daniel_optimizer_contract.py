"""Fast contract tests for the deployable Daniel optimization loop.

These checks intentionally avoid loading TRIBE or making paid API calls.
"""

import pytest
from pydantic import ValidationError

from neural_echo import analysis, ingest
from neural_echo.generator import Chunk, Genome, repair_genome
from neural_echo.optimizer import (
    MAX_GENERATION_ATTEMPTS,
    MODEL_ID,
    PATIENCE,
    OptimizerRun,
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
