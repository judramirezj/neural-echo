.PHONY: smoke test api calibration

VENV=.venv/bin

# Dry-run end-to-end at zero API cost (brief §6): builds a tiny stub
# calibration bundle if needed, then runs the optimizer against a local
# reference clip using pre-rendered stub audio instead of real ElevenLabs
# calls, and against the real (already-cached) TRIBE model.
smoke:
	$(VENV)/python scripts/smoke_test.py

test:
	$(VENV)/python -m pytest tests/ -v

calibration:
	$(VENV)/python scripts/build_clip_library.py

api:
	$(VENV)/uvicorn services.api.main:app --reload --port 8000
