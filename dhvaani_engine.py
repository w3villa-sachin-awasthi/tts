"""DhVaani-0.5 zero-shot voice clone — full WAV; prompt cached once."""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from log_config import setup_logger

logger = setup_logger(__name__)

MODEL_NAME = "dhvaani"
VOICE_NAME = "ref_clone"
SAMPLE_RATE = 24000
HF_REPO = "ARTPARK-IISc/DhVaani-0.5"
NUM_STEP = int(os.environ.get("DHVAANI_NUM_STEP", "16"))

ROOT = Path("/data/dhvaani")
REF_DIR = ROOT / "ref"
REF_AUDIO = REF_DIR / "voice.wav"
REF_TEXT = REF_DIR / "prompt.txt"
HF_CACHE = ROOT / "hf-cache"


class DhVaaniEngine:
    """Load DhVaani once; keep prompt wav + transcript pinned for every synth."""

    def __init__(self, device: str | None = None) -> None:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        HF_CACHE.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        self._prompt_wav: Path | None = REF_AUDIO if REF_AUDIO.is_file() else None
        self._prompt_text = REF_TEXT.read_text(encoding="utf-8").strip() if REF_TEXT.is_file() else ""

        from transformers import AutoModel

        t0 = time.perf_counter()
        self.model = (
            AutoModel.from_pretrained(
                HF_REPO,
                trust_remote_code=True,
                cache_dir=str(HF_CACHE),
            )
            .to(self.device)
            .eval()
        )
        sr = getattr(self.model, "sampling_rate", None) or getattr(self.model, "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)
        logger.info(
            "DhVaani ready device=%s load_s=%.1f ref=%s prompt_chars=%d num_step=%d",
            self.device,
            time.perf_counter() - t0,
            self._prompt_wav,
            len(self._prompt_text),
            NUM_STEP,
        )

    def set_reference(self, wav_path: Path, prompt_text: str | None = None) -> None:
        if not wav_path.is_file():
            raise FileNotFoundError(f"Reference WAV missing: {wav_path}")
        self._prompt_wav = wav_path
        if prompt_text is not None:
            text = prompt_text.strip()
            if not text:
                raise ValueError("prompt_text is required for DhVaani voice clone")
            REF_TEXT.write_text(text + "\n", encoding="utf-8")
            self._prompt_text = text
        elif REF_TEXT.is_file():
            self._prompt_text = REF_TEXT.read_text(encoding="utf-8").strip()
        if not self._prompt_text:
            raise ValueError(f"Write transcript to {REF_TEXT} (exact words of the ref WAV)")
        logger.info("DhVaani reference set wav=%s prompt_chars=%d", wav_path, len(self._prompt_text))

    def warm(self, _voice: str | None = None) -> None:
        if self._prompt_wav is None and REF_AUDIO.is_file():
            self._prompt_wav = REF_AUDIO
        if not self._prompt_text and REF_TEXT.is_file():
            self._prompt_text = REF_TEXT.read_text(encoding="utf-8").strip()

    def synthesize_array(self, text: str, language: str = "hi") -> np.ndarray:
        _ = language  # script is in the text; kept for API parity
        if self._prompt_wav is None or not Path(self._prompt_wav).is_file():
            raise FileNotFoundError(
                f"Reference WAV required at {REF_AUDIO}. "
                "Upload via POST /api/dhvaani/voice or scp a WAV there."
            )
        if not self._prompt_text:
            raise ValueError(
                f"Prompt transcript required at {REF_TEXT} "
                "(exact words spoken in the reference WAV)."
            )
        t0 = time.perf_counter()
        with self.lock:
            with torch.inference_mode():
                audio = self.model.synthesize(
                    text=text,
                    prompt_wav=str(self._prompt_wav),
                    prompt_text=self._prompt_text,
                    num_step=NUM_STEP,
                )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        wall = time.perf_counter() - t0
        audio_s = audio.size / float(self.sample_rate) if audio.size else 0.0
        logger.info(
            "DhVaani full-gen chars=%d audio_s=%.2f wall_s=%.2f rtf=%.2f",
            len(text),
            audio_s,
            wall,
            (wall / audio_s) if audio_s > 0 else 0.0,
        )
        return audio

    def synthesize(self, text: str, language: str = "hi") -> bytes:
        audio = self.synthesize_array(text, language=language)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
