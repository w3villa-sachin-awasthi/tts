#!/usr/bin/env python3
"""Indic Parler-TTS only. Does not import other TTS engines."""

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

from indic_parler_engine import SPEAKERS, IndicParlerEngine
from server_base import SingleModelHandler
from ui import render_page

engine = IndicParlerEngine()

LANG_CATALOG = [
    {
        "code": code,
        "name": meta["name"],
        "voices": [{"id": key, "label": label} for key, label in meta["voices"]],
    }
    for code, meta in SPEAKERS.items()
]

MODELS = [
    {
        "id": "indic-parler",
        "name": "Indic Parler-TTS",
        "route": "/api/indic-parler",
        "note": "Indic Parler only — other models stay unloaded. Hindi female: Divya / Rani.",
        "languages": LANG_CATALOG,
    },
]

LANG_SAMPLES = {
    "hi": "नमस्ते, मैं दिव्या हूँ। आज आप कैसे हैं?",
    "en": "Hello, this is Indic Parler-TTS on this Mac.",
    "bn": "নমস্কার, আপনি কেমন আছেন?",
    "mr": "नमस्कार, तुम्ही कसे आहात?",
    "gu": "નમસ્તે, તમે કેમ છો?",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
    "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "te": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    "kn": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    "ml": "നമസ്കാരം, സുഖമാണോ?",
    "or": "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?",
    "as": "নমস্কাৰ, আপুনি কেনে আছে?",
    "ne": "नमस्ते, तपाईं कस्तो हुनुहुन्छ?",
}


def synthesize(payload: dict[str, Any]) -> bytes:
    voice = str(payload.get("voice") or "Divya")
    lang = str(payload.get("language") or "hi")
    return engine.synthesize(str(payload["input"]).strip(), voice=voice, language=lang)


class Handler(SingleModelHandler):
    page_bytes = render_page(
        page_title="Indic Parler-TTS",
        models=MODELS,
        initial="indic-parler",
        sample_text=LANG_SAMPLES["hi"],
        loading_message="Synthesizing (first run loads Indic Parler)…",
        language_samples=LANG_SAMPLES,
    )
    page_paths = ("/", "/index.html", "/indic-parler")
    api_models = {
        "device": engine.device,
        "models": [
            {
                "id": "indic-parler",
                "repo": "ai4bharat/indic-parler-tts",
                "route": "/api/indic-parler",
                "sample_rate": 24000,
                "languages": LANG_CATALOG,
            }
        ],
    }
    route = "/api/indic-parler"
    synthesize_fn = staticmethod(synthesize)
    wav_name = "indic_parler.wav"
    output_dir = OUTPUT_DIR


def main() -> None:
    print("Indic Parler-TTS only (other models not loaded).", flush=True)
    print(f"Open http://127.0.0.1:{UI_PORT}/", flush=True)
    print("Default: Hindi + Divya (female).", flush=True)
    ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
