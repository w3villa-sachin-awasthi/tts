"""Hardcoded Qwen3-TTS settings — edit here; no .env.

Restart: pm2 restart tts-qwen3
To remove this stack later: delete tts/qwen3-tts + pm2 delete tts-qwen3.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Switch / pin here ──────────────────────────────────────────────
TTS_HOST = "127.0.0.1"
TTS_PORT = 8766

# HF id OR local folder with Base weights (0.6B for Jetson):
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
# MODEL_ID = str(ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base")

VOICE_NAME = "clone"
LANGUAGE_NAME = "Auto"  # Auto | English | Hindi
SAMPLE_RATE = 24000

REF_PATH = ROOT / "ref" / "speaker.wav"
PROMPT_PATH = ROOT / "cache" / "voice_clone_prompt.pt"

# True = speaker embed only (no transcript). False = ICL (needs REF_TEXT).
X_VECTOR_ONLY = True
REF_TEXT = ""
# REF_TEXT = "exact transcript of speaker.wav"
# X_VECTOR_ONLY = False

ROUTE = "/api/qwen3"
MODEL_NAME = "qwen3"
