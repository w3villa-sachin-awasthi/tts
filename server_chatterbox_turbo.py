#!/usr/bin/env python3
"""Chatterbox-Turbo only. Does not import other TTS engines."""

from __future__ import annotations

import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "huggingface"
OUTPUT_DIR = ROOT / "output"
UI_PORT = 8765

os.environ.setdefault("HF_HOME", str(MODEL_DIR))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from chatterbox_turbo_engine import LANGUAGES as TURBO_LANGUAGES
from chatterbox_turbo_engine import VOICES as TURBO_VOICES
from chatterbox_turbo_engine import ChatterboxTurboEngine
from server_base import SingleModelHandler
from ui import render_page

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
engine = ChatterboxTurboEngine(DEVICE)

TURBO_LANG = [
    {
        "code": code,
        "name": name,
        "voices": [{"id": key, "label": label} for key, label in TURBO_VOICES],
    }
    for code, name in TURBO_LANGUAGES.items()
]

MODELS = [
    {
        "id": "chatterbox-turbo",
        "name": "Chatterbox-Turbo",
        "route": "/api/chatterbox-turbo",
        "note": (
            "Turbo only (English, 350M). Tags like [laugh]/[chuckle] work in text. "
            "Optional ref: voices/turbo_default.wav (>5s). Other models stay unloaded."
        ),
        "languages": TURBO_LANG,
    },
]


def synthesize(payload: dict[str, Any]) -> bytes:
    voice = str(payload.get("voice") or "default")
    lang = str(payload.get("language") or "en")
    return engine.synthesize(str(payload["input"]).strip(), voice, language=lang)


class Handler(SingleModelHandler):
    page_bytes = render_page(
        page_title="Chatterbox-Turbo",
        models=MODELS,
        initial="chatterbox-turbo",
        sample_text=(
            "Hi there, calling you back [chuckle], have you got one minute "
            "to chat about the billing issue?"
        ),
        loading_message="Synthesizing (first run loads Chatterbox-Turbo)…",
    )
    page_paths = ("/", "/index.html", "/chatterbox-turbo")
    api_models = {
        "device": DEVICE,
        "models": [
            {
                "id": "chatterbox-turbo",
                "repo": "ResembleAI/chatterbox-turbo",
                "route": "/api/chatterbox-turbo",
                "sample_rate": 24000,
                "languages": TURBO_LANG,
            }
        ],
    }
    route = "/api/chatterbox-turbo"
    synthesize_fn = staticmethod(synthesize)
    wav_name = "chatterbox_turbo.wav"
    output_dir = OUTPUT_DIR


def main() -> None:
    print("Chatterbox-Turbo only (other models not loaded).", flush=True)
    print(f"Open http://127.0.0.1:{UI_PORT}/", flush=True)
    print(f"Device: {DEVICE}. First Speak loads ResembleAI/chatterbox-turbo.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
