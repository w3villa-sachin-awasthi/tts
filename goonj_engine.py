"""Goonj-1-82M (Kokoro Indian EN/HI fine-tune) — true streaming via KPipeline generator."""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from log_config import setup_logger

logger = setup_logger(__name__)

SAMPLE_RATE = 24000

# Persona voicepacks shipped with BH-Builds/goonj-1-82M
GOONJ_VOICES: tuple[str, ...] = (
    "en_aman",
    "en_ananya",
    "en_arjun",
    "en_dev",
    "en_divya",
    "en_kabir",
    "en_nisha",
    "en_priya",
    "en_sameer",
    "en_tara",
    "hi_atul",
    "hi_meera",
    "hi_ravi",
    "hi_shivani",
    "bed_hindi",
)

# English personas use US English G2P (lang_code "a"); Hindi use "h".
VOICE_LANG = {
    **{v: "a" for v in GOONJ_VOICES if v.startswith("en_")},
    **{v: "h" for v in GOONJ_VOICES if v.startswith("hi_") or v == "bed_hindi"},
}


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def default_model_dir() -> Path:
    raw = _env("GOONJ_MODEL_DIR")
    if raw:
        return Path(raw)
    # Jetson layout used in trials; fall back to repo-local models/
    for candidate in (
        Path("/data/goonj/models/goonj-1-82M"),
        Path(__file__).resolve().parent / "models" / "goonj-1-82M",
    ):
        if (candidate / "kokoro_hindi_final.pth").is_file():
            return candidate
    return Path("/data/goonj/models/goonj-1-82M")


class GoonjEngine:
    """One Goonj checkpoint + one voice tensor in RAM (ACTIVE_TTS single-engine policy)."""

    def __init__(self, device: str | None = None) -> None:
        from kokoro import KModel, KPipeline

        self.root = default_model_dir()
        config = self.root / "config.json"
        weights = self.root / "kokoro_hindi_final.pth"
        if not config.is_file() or not weights.is_file():
            raise FileNotFoundError(
                f"Goonj model not found under {self.root}. "
                "Set GOONJ_MODEL_DIR to the goonj-1-82M folder "
                "(needs config.json + kokoro_hindi_final.pth + voices/)."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        self._voice_name: str | None = None
        self._voice_pack = None
        self._pipelines: dict[str, object] = {}

        self.model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(config),
            model=str(weights),
        ).to(self.device).eval()
        self._KPipeline = KPipeline
        logger.info("Goonj loaded root=%s device=%s", self.root, self.device)

    def _pipeline(self, lang: str):
        if lang not in self._pipelines:
            self._pipelines[lang] = self._KPipeline(
                lang_code=lang,
                repo_id="hexgrad/Kokoro-82M",
                model=self.model,
            )
        return self._pipelines[lang]

    def _pack_for(self, voice: str):
        if voice not in VOICE_LANG:
            raise ValueError(f"Unknown Goonj voice {voice!r}. Use one of: {', '.join(GOONJ_VOICES)}")
        if self._voice_name == voice and self._voice_pack is not None:
            return self._voice_pack
        path = self.root / "voices" / f"{voice}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing voicepack: {path}")
        self._voice_pack = torch.load(path, map_location="cpu", weights_only=True)
        self._voice_name = voice
        logger.info("Goonj voice loaded %s", voice)
        return self._voice_pack

    def warm(self, voice: str) -> None:
        with self.lock:
            self._pack_for(voice)
            lang = VOICE_LANG[voice]
            self._pipeline(lang)

    def iter_audio(self, text: str, voice: str, speed: float = 1.0):
        lang = VOICE_LANG.get(voice)
        if lang is None:
            raise ValueError(f"Unknown Goonj voice {voice!r}")
        with self.lock:
            pack = self._pack_for(voice)
            pipeline = self._pipeline(lang)
            for _gs, _ps, audio in pipeline(text, voice=pack, speed=speed):
                if audio is None:
                    continue
                chunk = np.asarray(audio.detach().cpu(), dtype=np.float32).reshape(-1)
                if chunk.size:
                    yield chunk

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        parts = list(self.iter_audio(text, voice, speed=speed))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
