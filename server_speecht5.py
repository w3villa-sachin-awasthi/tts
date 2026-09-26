#!/usr/bin/env python3
"""SpeechT5 only. Does not import other TTS engines."""

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

from server_base import SingleModelHandler
from speecht5_engine import SPEAKERS, SpeechT5Engine
from ui import render_page

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
engine = SpeechT5Engine(DEVICE)

SPEECHT5_LANG = [
    {
        "code": "en",
        "name": "English",
        "voices": [{"id": key, "label": label} for key, label in SPEAKERS],
    }
]

MODELS = [
    {
        "id": "speecht5",
        "name": "SpeechT5",
        "route": "/api/speecht5",
        "note": "SpeechT5 only — other models stay unloaded. English / LibriTTS speakers.",
        "languages": SPEECHT5_LANG,
    },
]


def synthesize(payload: dict[str, Any]) -> bytes:
    voice = str(payload.get("voice") or "slt")
    return engine.synthesize(str(payload["input"]).strip(), voice)


class Handler(SingleModelHandler):
    page_bytes = render_page(
        page_title="SpeechT5",
        models=MODELS,
        initial="speecht5",
        sample_text="Hello from SpeechT5 on this Mac.",
        loading_message="Synthesizing (first run loads SpeechT5)…",
    )
    page_paths = ("/", "/index.html", "/speecht5")
    api_models = {
        "device": DEVICE,
        "models": [
            {
                "id": "speecht5",
                "repo": "microsoft/speecht5_tts",
                "route": "/api/speecht5",
                "sample_rate": 16000,
                "languages": SPEECHT5_LANG,
            }
        ],
    }
    route = "/api/speecht5"
    synthesize_fn = staticmethod(synthesize)
    wav_name = "speecht5.wav"
    output_dir = OUTPUT_DIR


def main() -> None:
    print("SpeechT5 only (other models not loaded).", flush=True)
    print(f"Open http://127.0.0.1:{UI_PORT}/", flush=True)
    print(f"Device: {DEVICE}. First Speak loads weights.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
