#!/usr/bin/env python3
"""Chatterbox Multilingual only. Does not import other TTS engines."""

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

from chatterbox_engine import LANGUAGES as CHATTERBOX_LANGUAGES
from chatterbox_engine import VOICES as CHATTERBOX_VOICES
from chatterbox_engine import ChatterboxEngine
from server_base import SingleModelHandler
from ui import render_page

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
engine = ChatterboxEngine(DEVICE)

CHATTERBOX_LANG = [
    {
        "code": code,
        "name": name,
        "voices": [{"id": key, "label": label} for key, label in CHATTERBOX_VOICES],
    }
    for code, name in CHATTERBOX_LANGUAGES.items()
]

MODELS = [
    {
        "id": "chatterbox",
        "name": "Chatterbox Multilingual",
        "route": "/api/chatterbox",
        "note": "Chatterbox only — other models stay unloaded. First Speak loads weights.",
        "languages": CHATTERBOX_LANG,
    },
]


def synthesize(payload: dict[str, Any]) -> bytes:
    voice = str(payload.get("voice") or "default")
    lang = str(payload.get("language") or "en")
    return engine.synthesize(str(payload["input"]).strip(), voice, language=lang)


class Handler(SingleModelHandler):
    page_bytes = render_page(
        page_title="Chatterbox",
        models=MODELS,
        initial="chatterbox",
        sample_text="Hello from Chatterbox on this Mac.",
        loading_message="Synthesizing (first run loads Chatterbox)…",
    )
    page_paths = ("/", "/index.html", "/chatterbox")
    api_models = {
        "device": DEVICE,
        "models": [
            {
                "id": "chatterbox",
                "repo": "ResembleAI/chatterbox",
                "route": "/api/chatterbox",
                "sample_rate": 24000,
                "languages": CHATTERBOX_LANG,
            }
        ],
    }
    route = "/api/chatterbox"
    synthesize_fn = staticmethod(synthesize)
    wav_name = "chatterbox.wav"
    output_dir = OUTPUT_DIR


def main() -> None:
    print("Chatterbox only (other models not loaded).", flush=True)
    print(f"Open http://127.0.0.1:{UI_PORT}/", flush=True)
    print(f"Device: {DEVICE}. First Speak loads weights.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
