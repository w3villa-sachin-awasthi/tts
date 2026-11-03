"""Single-active TTS engine policy.

AIAA-173: model/voice/language are hardcoded (no .env). Revert by checking out
the previous branch.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

from log_config import setup_logger

logger = setup_logger(__name__)

KNOWN_MODELS = ("kokoro", "goonj", "qwen", "chatterbox", "dhvaani")

ROUTE_TO_MODEL = {
    "/api/speech": "kokoro",
    "/api/goonj": "goonj",
    "/api/qwen": "qwen",
    "/api/chatterbox": "chatterbox",
    "/api/dhvaani": "dhvaani",
}

MODEL_TO_ROUTE = {model: route for route, model in ROUTE_TO_MODEL.items()}

MODEL_SAMPLE_RATE = {
    "kokoro": 24000,
    "goonj": 24000,
    "qwen": 24000,
    "chatterbox": 24000,
    "dhvaani": 24000,
}

# ── Switch TTS engine here (no ACTIVE_TTS_* .env). Restart: pm2 restart tts ──
# Keep only ONE block active; comment the others.
# Parallel PM2: leave goonj here; start tts-dhvaani with TTS_MODEL_NAME=dhvaani.
# MODEL_NAME = "qwen"
# VOICE_NAME = "hr_clone"
# LANGUAGE_NAME = "English"

# MODEL_NAME = "chatterbox"
# VOICE_NAME = "ref_clone"
# LANGUAGE_NAME = ""  # en / hi via API

# MODEL_NAME = "dhvaani"
# VOICE_NAME = "ref_clone"
# LANGUAGE_NAME = "hi"

MODEL_NAME = "goonj"
VOICE_NAME = "en_priya"
LANGUAGE_NAME = "a"

# MODEL_NAME = "kokoro"
# VOICE_NAME = "af_heart"
# LANGUAGE_NAME = "a"

# Optional override for a second PM2 process (does not touch goonj service).
_env_model = os.environ.get("TTS_MODEL_NAME", "").strip()
if _env_model:
    MODEL_NAME = _env_model
    if MODEL_NAME == "dhvaani":
        VOICE_NAME = os.environ.get("TTS_VOICE_NAME", "ref_clone").strip() or "ref_clone"
        LANGUAGE_NAME = os.environ.get("TTS_LANGUAGE_NAME", "hi").strip() or "hi"

MODEL_ROUTE = MODEL_TO_ROUTE[MODEL_NAME]
SAMPLE_RATE = MODEL_SAMPLE_RATE[MODEL_NAME]

if MODEL_NAME not in KNOWN_MODELS:
    raise SystemExit(
        f"MODEL_NAME={MODEL_NAME!r} is invalid. Use one of: {', '.join(KNOWN_MODELS)}"
    )

ACTIVE_TTS_MODEL = MODEL_NAME
ACTIVE_TTS_VOICE = VOICE_NAME
ACTIVE_TTS_LANGUAGE = LANGUAGE_NAME

logger.info(
    "Active TTS hardcoded model_name=%s voice_name=%s language_name=%s route=%s",
    MODEL_NAME,
    VOICE_NAME,
    LANGUAGE_NAME,
    MODEL_ROUTE,
)


def inactive_message(requested: str) -> str:
    return (
        f"Engine {requested!r} is not active. "
        f"MODEL_NAME={ACTIVE_TTS_MODEL!r}. "
        f"Only the active engine is in RAM on this branch."
    )


def ensure_route_active(path: str) -> str | None:
    """Return an error message if path's engine is not the active one."""
    model = ROUTE_TO_MODEL.get(path)
    if model is None:
        return None
    if model != ACTIVE_TTS_MODEL:
        return inactive_message(model)
    return None


def pin_voice_error(voice: str) -> str | None:
    if ACTIVE_TTS_VOICE and voice != ACTIVE_TTS_VOICE:
        return (
            f"Voice {voice!r} is not the pinned VOICE_NAME={ACTIVE_TTS_VOICE!r}."
        )
    return None


def pin_language_error(lang: str) -> str | None:
    if ACTIVE_TTS_LANGUAGE and lang != ACTIVE_TTS_LANGUAGE:
        return (
            f"Language {lang!r} is not the pinned LANGUAGE_NAME={ACTIVE_TTS_LANGUAGE!r}."
        )
    return None


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def default_voice_for(model: str) -> str:
    if ACTIVE_TTS_VOICE:
        return ACTIVE_TTS_VOICE
    return {
        "kokoro": "af_heart",
        "goonj": "en_priya",
        "qwen": "hr_clone",
        "chatterbox": "ref_clone",
        "dhvaani": "ref_clone",
    }[model]


def default_language_for(model: str) -> str:
    if ACTIVE_TTS_LANGUAGE:
        return ACTIVE_TTS_LANGUAGE
    return {
        "kokoro": "a",
        "goonj": "a",
        "qwen": "English",
        "chatterbox": "en",
        "dhvaani": "hi",
    }[model]


def write_env_hint(root: Path) -> str:
    return (
        f"MODEL_NAME={ACTIVE_TTS_MODEL} "
        f"VOICE_NAME={ACTIVE_TTS_VOICE} "
        f"LANGUAGE_NAME={ACTIVE_TTS_LANGUAGE} "
        f"MODEL_ROUTE={MODEL_ROUTE} "
        f"SAMPLE_RATE={SAMPLE_RATE} "
        f"models_dir={root / 'models'}"
    )
