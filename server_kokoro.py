#!/usr/bin/env python3
"""Kokoro-82M only. Does not import other TTS engines."""

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

from kokoro_engine import KokoroEngine, build_catalog
from server_base import SingleModelHandler
from ui import render_page

engine = KokoroEngine()
CATALOG = build_catalog()
VOICE_LANG = {
    voice["id"]: item["code"] for item in CATALOG for voice in item["voices"]
}

MODELS = [
    {
        "id": "kokoro",
        "name": "Kokoro-82M",
        "route": "/api/speech",
        "note": "Kokoro only — other models stay unloaded. Voice must match language.",
        "languages": CATALOG,
    },
]


def synthesize(payload: dict[str, Any]) -> bytes:
    voice = str(payload.get("voice") or "af_heart")
    lang = str(payload.get("language") or VOICE_LANG.get(voice, "a"))
    if voice not in VOICE_LANG or VOICE_LANG[voice] != lang:
        raise ValueError("That voice does not match the selected language.")
    return engine.synthesize(str(payload["input"]).strip(), voice, lang)


class Handler(SingleModelHandler):
    page_bytes = render_page(
        page_title="Kokoro-82M",
        models=MODELS,
        initial="kokoro",
        sample_text="Hello from Kokoro on this Mac.",
        loading_message="Synthesizing (first run loads Kokoro)…",
    )
    page_paths = ("/", "/index.html", "/kokoro")
    api_models = {
        "device": engine.device,
        "models": [
            {
                "id": "kokoro",
                "repo": "hexgrad/Kokoro-82M",
                "route": "/api/speech",
                "sample_rate": 24000,
                "languages": CATALOG,
            }
        ],
    }
    route = "/api/speech"
    synthesize_fn = staticmethod(synthesize)
    wav_name = "kokoro.wav"
    output_dir = OUTPUT_DIR


def main() -> None:
    print("Kokoro-82M only (other models not loaded).", flush=True)
    print(f"Open http://127.0.0.1:{UI_PORT}/", flush=True)
    print(f"Device: {engine.device}. First Speak loads weights.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
