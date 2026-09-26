#!/usr/bin/env python3
"""CosyVoice2 only. Does not import other TTS engines."""

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

from cosyvoice_engine import LANGUAGES, VOICES, CosyVoiceEngine
from server_base import SingleModelHandler
from ui import render_page

engine = CosyVoiceEngine()

COSY_LANG = [
    {
        "code": code,
        "name": name,
        "voices": [{"id": key, "label": label} for key, label in VOICES],
    }
    for code, name in LANGUAGES.items()
]

MODELS = [
    {
        "id": "cosyvoice",
        "name": "CosyVoice2-0.5B",
        "route": "/api/cosyvoice",
        "note": "CosyVoice2 only — other models stay unloaded. First Speak loads weights.",
        "languages": COSY_LANG,
    },
]


def synthesize(payload: dict[str, Any]) -> bytes:
    voice = str(payload.get("voice") or "default")
    lang = str(payload.get("language") or "en")
    return engine.synthesize(str(payload["input"]).strip(), language=lang, voice=voice)


class Handler(SingleModelHandler):
    page_bytes = render_page(
        page_title="CosyVoice2-0.5B",
        models=MODELS,
        initial="cosyvoice",
        sample_text="Hello from CosyVoice2 on this Mac.",
        loading_message="Synthesizing (first run loads CosyVoice2)…",
    )
    page_paths = ("/", "/index.html", "/cosyvoice")
    api_models = {
        "device": "cpu",
        "models": [
            {
                "id": "cosyvoice",
                "repo": "FunAudioLLM/CosyVoice2-0.5B",
                "route": "/api/cosyvoice",
                "sample_rate": 24000,
                "languages": COSY_LANG,
            }
        ],
    }
    route = "/api/cosyvoice"
    synthesize_fn = staticmethod(synthesize)
    wav_name = "cosyvoice.wav"
    output_dir = OUTPUT_DIR


def main() -> None:
    print("CosyVoice2 only (other models not loaded).", flush=True)
    print(f"Open http://127.0.0.1:{UI_PORT}/", flush=True)
    print("First Speak loads CosyVoice2 weights into memory.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
