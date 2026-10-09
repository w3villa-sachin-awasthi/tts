"""Single-active TTS engine policy.

Models may all remain on disk under ./models. Only ACTIVE_TTS_MODEL is loaded into RAM.
Switch engines by changing env and restarting the TTS process (e.g. pm2 restart tts --update-env).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

KNOWN_MODELS = ("kokoro", "piper", "speecht5", "magpie", "parler")

ROUTE_TO_MODEL = {
    "/api/speech": "kokoro",
    "/api/piper": "piper",
    "/api/speecht5": "speecht5",
    "/api/magpie": "magpie",
    "/api/parler": "parler",
}

MODEL_TO_ROUTE = {model: route for route, model in ROUTE_TO_MODEL.items()}

MODEL_SAMPLE_RATE = {
    "kokoro": 24000,
    "piper": 22050,
    "speecht5": 16000,
    "magpie": 22050,
    "parler": 44100,
}


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def resolve_active_model() -> str:
    raw = _env("ACTIVE_TTS_MODEL", "piper").lower()
    if raw not in KNOWN_MODELS:
        raise SystemExit(
            f"ACTIVE_TTS_MODEL={raw!r} is invalid. Use one of: {', '.join(KNOWN_MODELS)}"
        )
    return raw


ACTIVE_TTS_MODEL = resolve_active_model()
ACTIVE_TTS_VOICE = _env("ACTIVE_TTS_VOICE")
ACTIVE_TTS_LANGUAGE = _env("ACTIVE_TTS_LANGUAGE")


def inactive_message(requested: str) -> str:
    return (
        f"Engine {requested!r} is not active. "
        f"ACTIVE_TTS_MODEL={ACTIVE_TTS_MODEL!r}. "
        f"Models stay on disk; only the active engine is in RAM. "
        f"Set ACTIVE_TTS_MODEL={requested} (and matching voice/language), "
        f"then restart TTS: pm2 restart tts --update-env"
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
            f"Voice {voice!r} is not the pinned ACTIVE_TTS_VOICE={ACTIVE_TTS_VOICE!r}. "
            "Change ACTIVE_TTS_VOICE and restart TTS to switch (keeps one voice in RAM)."
        )
    return None


def pin_language_error(lang: str) -> str | None:
    if ACTIVE_TTS_LANGUAGE and lang != ACTIVE_TTS_LANGUAGE:
        return (
            f"Language {lang!r} is not the pinned ACTIVE_TTS_LANGUAGE={ACTIVE_TTS_LANGUAGE!r}. "
            "Change ACTIVE_TTS_LANGUAGE and restart TTS to switch."
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
        "piper": "en_US-lessac-medium",
        "speecht5": "slt",
        "magpie": "Sofia",
        "parler": "Mary",
    }[model]


def default_language_for(model: str) -> str:
    if ACTIVE_TTS_LANGUAGE:
        return ACTIVE_TTS_LANGUAGE
    return {
        "kokoro": "a",
        "piper": "en",
        "speecht5": "en",
        "magpie": "en-US",
        "parler": "en",
    }[model]


def write_env_hint(root: Path) -> str:
    return (
        f"ACTIVE_TTS_MODEL={ACTIVE_TTS_MODEL} "
        f"ACTIVE_TTS_VOICE={ACTIVE_TTS_VOICE or '(any for this engine)'} "
        f"ACTIVE_TTS_LANGUAGE={ACTIVE_TTS_LANGUAGE or '(any for this engine)'} "
        f"models_dir={root / 'models'}"
    )
