"""Chatterbox Multilingual V3 (500M) voice-clone — full-waveform only (no incremental decode)."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from log_config import setup_logger

logger = setup_logger(__name__)

MODEL_NAME = "chatterbox"
VOICE_NAME = "ref_clone"
SAMPLE_RATE = 24000

# Layout on Jetson (keep Qwen / Goonj paths untouched)
ROOT = Path("/data/chatterbox")
REF_DIR = ROOT / "ref"
REF_AUDIO = REF_DIR / "voice.wav"
HF_CACHE = ROOT / "hf-cache"

class ChatterboxEngine:
    """One Multilingual V3 checkpoint + one reference WAV (single-flight lock)."""

    def __init__(self, device: str | None = None) -> None:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        HF_CACHE.mkdir(parents=True, exist_ok=True)
        # Keep HF downloads off the root partition.
        import os

        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        self._ref_path = REF_AUDIO if REF_AUDIO.is_file() else None

        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        t0 = time.perf_counter()
        self.model = ChatterboxMultilingualTTS.from_pretrained(
            device=self.device,
            t3_model="v3",
        )
        self.sample_rate = int(getattr(self.model, "sr", SAMPLE_RATE) or SAMPLE_RATE)
        logger.info(
            "Chatterbox Multilingual V3 ready device=%s load_s=%.1f ref=%s",
            self.device,
            time.perf_counter() - t0,
            self._ref_path,
        )

    def set_reference(self, wav_path: Path) -> None:
        if not wav_path.is_file():
            raise FileNotFoundError(f"Reference WAV missing: {wav_path}")
        self._ref_path = wav_path
        logger.info("Chatterbox reference set %s", wav_path)

    def warm(self, _voice: str | None = None) -> None:
        if self._ref_path is None and REF_AUDIO.is_file():
            self._ref_path = REF_AUDIO

    def synthesize_array(self, text: str, language: str = "en") -> np.ndarray:
        lang = (language or "en").strip().lower()
        if lang in ("english", "a"):
            lang = "en"
        if lang in ("hindi", "h", "hi-in"):
            lang = "hi"
        if self._ref_path is None or not Path(self._ref_path).is_file():
            raise FileNotFoundError(
                f"Reference WAV required at {REF_AUDIO}. "
                "Upload via POST /api/chatterbox/voice or scp a WAV there."
            )
        t0 = time.perf_counter()
        with self.lock:
            wav = self.model.generate(
                text,
                language_id=lang,
                audio_prompt_path=str(self._ref_path),
            )
        if torch.is_tensor(wav):
            audio = wav.detach().float().cpu().numpy().reshape(-1)
        else:
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        wall = time.perf_counter() - t0
        audio_s = audio.size / float(self.sample_rate) if audio.size else 0.0
        logger.info(
            "Chatterbox full-gen lang=%s chars=%d audio_s=%.2f wall_s=%.2f rtf=%.2f",
            lang,
            len(text),
            audio_s,
            wall,
            (wall / audio_s) if audio_s > 0 else 0.0,
        )
        return audio

    def synthesize(self, text: str, language: str = "en") -> bytes:
        audio = self.synthesize_array(text, language=language)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
