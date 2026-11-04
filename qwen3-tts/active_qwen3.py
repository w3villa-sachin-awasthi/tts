"""Hardcoded Qwen3-TTS TRT settings — edit here; no .env.

Restart: pm2 restart tts-qwen3
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Server ─────────────────────────────────────────────────────────
TTS_HOST = "127.0.0.1"
TTS_PORT = 8766
ROUTE = "/api/qwen3"
MODEL_NAME = "qwen3"
VOICE_NAME = "clone"
LANGUAGE_NAME = "English"
SAMPLE_RATE = 24000

# ── Paths (Jetson after TRT download) ──────────────────────────────
TRT_ROOT = ROOT / "trt" / "qwen3-tts-base-trt"

WORKER = TRT_ROOT / "jv-workers" / "qwen3_tts_streaming_worker"
PLUGIN = TRT_ROOT / "jv-workers" / "libNvInfer_edgellm_plugin.so"

MODEL_DIR = TRT_ROOT / "models" / "qwen3-tts-base"
TALKER_DIR = MODEL_DIR / "engines" / "talker"
CP_DIR = MODEL_DIR / "engines" / "code_predictor"
CODE2WAV_DIR = MODEL_DIR / "engines" / "code2wav"
TOKENIZER_DIR = TALKER_DIR

SPEAKER_ENCODER_ONNX = MODEL_DIR / "onnx" / "speaker_encoder.onnx"
EXTRACT_SCRIPT = TRT_ROOT / "scripts" / "extract_speaker_embedding.py"

REF_PATH = ROOT / "ref" / "speaker.wav"
EMBED_PATH = ROOT / "cache" / "speaker_emb.b64"

# TRT generate once (no mid-utt vocoder splits), then SSE in even pieces.
# stream=False → smoother audio; SSE_OUT_CHUNK = ~200ms @ 24kHz for LiveKit.
TRT_STREAM = False
SSE_OUT_CHUNK = 4800
TALKER_TEMPERATURE = 0.7
# Used only if TRT_STREAM=True (legacy):
FIRST_CHUNK_FRAMES = 6
CHUNK_FRAMES = 10
