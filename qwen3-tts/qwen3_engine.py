"""Qwen3-TTS Base via Jetson TensorRT worker + cached speaker embed.

Default: full utterance from TRT, then even SSE chunks (smooth playback).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.append(str(_PARENT))

from log_config import setup_logger

from active_qwen3 import (
    CHUNK_FRAMES,
    CODE2WAV_DIR,
    CP_DIR,
    EMBED_PATH,
    FIRST_CHUNK_FRAMES,
    LANGUAGE_NAME,
    PLUGIN,
    REF_PATH,
    SAMPLE_RATE,
    SSE_OUT_CHUNK,
    TALKER_DIR,
    TALKER_TEMPERATURE,
    TOKENIZER_DIR,
    TRT_STREAM,
    WORKER,
)

logger = setup_logger(__name__)

LANG_MAP = {
    "a": "English",
    "en": "English",
    "english": "English",
    "h": "Hindi",
    "hi": "Hindi",
    "hindi": "Hindi",
    "auto": "Auto",
}


def default_ref_path() -> Path:
    return Path(REF_PATH)


def default_embed_path() -> Path:
    return Path(EMBED_PATH)


def normalize_language(lang: str | None) -> str:
    if not lang:
        return LANGUAGE_NAME
    return LANG_MAP.get(lang.strip().lower(), lang.strip())


class Qwen3Engine:
    """Long-lived TRT worker; speaker_emb.b64 loaded once."""

    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE
        self.device = "trt"
        self.model_id = str(TALKER_DIR)
        self.prompt_path = default_embed_path()
        self.lock = threading.Lock()
        self._emb_b64: str | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._start_worker()

    def _start_worker(self) -> None:
        if not WORKER.is_file():
            raise FileNotFoundError(
                f"TRT worker missing: {WORKER}\n"
                "Download Base Jetson TRT bundle into trt/qwen3-tts-base-trt/"
            )
        for p in (TALKER_DIR, CP_DIR, CODE2WAV_DIR, TOKENIZER_DIR, PLUGIN):
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing TRT path: {p}")

        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = str(PLUGIN)
        cuda = "/usr/local/cuda/lib64"
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda}:{prev}" if prev else cuda

        cmd = [
            str(WORKER),
            f"--talkerEngineDir={TALKER_DIR}",
            f"--codePredictorEngineDir={CP_DIR}",
            f"--code2wavEngineDir={CODE2WAV_DIR}",
            f"--tokenizerDir={TOKENIZER_DIR}",
        ]
        logger.info("Starting TRT worker: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("TRT worker exited before ready")
            line = line.strip()
            logger.info("TRT worker: %s", line[:200])
            low = line.lower()
            if "ready" in low:
                break
            try:
                obj = json.loads(line)
                if obj.get("event") == "ready" or obj.get("status") == "ready":
                    break
            except json.JSONDecodeError:
                pass

    def load_embed(self, path: Path | None = None) -> str:
        path = Path(path or self.prompt_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing speaker embed: {path}. Run extract for ref/speaker.wav"
            )
        self._emb_b64 = path.read_text(encoding="utf-8").strip()
        self.prompt_path = path
        logger.info("Speaker embed loaded path=%s bytes=%d", path, len(self._emb_b64))
        return self._emb_b64

    def ensure_embed(self) -> str:
        if self._emb_b64:
            return self._emb_b64
        return self.load_embed()

    def warm(self) -> None:
        with self.lock:
            self.ensure_embed()

    def _read_pcm_until_done(self, proc: subprocess.Popen[str]) -> np.ndarray:
        assert proc.stdout is not None
        parts: list[np.ndarray] = []
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("TRT worker closed stdout mid-request")
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("TRT non-json: %s", line[:160])
                continue

            et = ev.get("event") or ev.get("type") or ""
            if et == "chunk":
                b64 = ev.get("audio_b64") or ev.get("b64") or ""
                if not b64:
                    continue
                pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                if pcm.size:
                    parts.append(pcm)
            elif et in ("done", "end"):
                break
            elif et == "error":
                raise RuntimeError(ev.get("message") or line)

        if not parts:
            return np.zeros(0, dtype=np.float32)
        wav = np.concatenate(parts).astype(np.float32) / 32768.0
        return wav

    def iter_audio(self, text: str, language: str | None = None) -> Iterator[np.ndarray]:
        text = (text or "").strip()
        if not text:
            return
        lang = normalize_language(language or LANGUAGE_NAME)
        with self.lock:
            emb = self.ensure_embed()
            proc = self._proc
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError("TRT worker not running")
            if proc.poll() is not None:
                raise RuntimeError(f"TRT worker died exit={proc.returncode}")

            req: dict = {
                "id": str(uuid.uuid4()),
                "text": text,
                "language": lang,
                "speaker_embedding_b64": emb,
                "stream": bool(TRT_STREAM),
                "talker_temperature": TALKER_TEMPERATURE,
                "chunk_format": "pcm_s16le",
                "chunk_transport": "base64",
            }
            if TRT_STREAM:
                req["chunk_frames"] = CHUNK_FRAMES
                req["first_chunk_frames"] = FIRST_CHUNK_FRAMES

            logger.info(
                "TRT synth stream=%s lang=%s chars=%d",
                TRT_STREAM,
                lang,
                len(text),
            )
            proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
            proc.stdin.flush()

            wav = self._read_pcm_until_done(proc)
            if wav.size == 0:
                return

            # Even SSE pieces for LiveKit (avoids bursty/gap playback).
            step = max(int(SSE_OUT_CHUNK), 1)
            for i in range(0, wav.size, step):
                piece = wav[i : i + step]
                if piece.size:
                    yield piece
