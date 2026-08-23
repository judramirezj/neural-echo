"""Fast checks for user-facing recovery messages."""

from services.api.jobs import _friendly_job_error


def test_composition_safety_error_is_more_specific_than_provider_error():
    error = RuntimeError("ElevenLabs bad_composition_plan: terms of service")

    message = _friendly_job_error(error)

    assert "safely rendered" in message
    assert "names or brands" in message


def test_stale_container_error_suggests_a_fresh_run():
    message = _friendly_job_error(RuntimeError("response from daemon: No such container: abc"))

    assert "restarted" in message
    assert "new run" in message
