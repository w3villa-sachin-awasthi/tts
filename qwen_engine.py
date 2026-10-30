"""Qwen3-TTS-12Hz-0.6B-Base voice-clone engine (Jetson-oriented).

Hardcoded paths/dtype for AIAA-173. No env. Revert by checking out previous branch.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from log_config import setup_logger

logger = setup_logger(__name__)

MODEL_NAME = "qwen"
VOICE_NAME = "hr_clone"
LANGUAGE_NAME = "English"
SAMPLE_RATE = 24000

MODEL_DIR = Path("/data/qwen3-tts/models/0.6B-Base")
REF_AUDIO = Path("/data/qwen3-tts/ref/HR_VOICE.wav")
DTYPE = torch.float32  # float16 often NaNs on Jetson Orin

# Exact transcript of REF_AUDIO (HR_VOICE.wav)
REF_TEXT = (
    "Hello, shradha! "
    "My name is srishti pandey, and I'll be your interviewer today. I hope you're doing well. "
    "Let's begin with a brief introduction. "
    "Could you please introduce yourself... and tell me a little about your educational background? "
    "Next, I'd like to learn more about your technical skills... your key strengths... "
    "and the areas where you feel you could improve "
    "And finally, could you walk me through some of the projects... or internships... "
    "that you have completed? "
    "Please explain your role... the technologies you worked with... "
    "and your key contributions to each project. "
    "Take your time... and feel free to answer in as much detail as you'd like."
)

# SSE chunk size (~20 ms @ 24 kHz) for LocalHttpTTS / telephony
_CHUNK_SAMPLES = int(SAMPLE_RATE * 0.020)


class QwenEngine:
    """One Qwen Base checkpoint + one cached voice-clone prompt in RAM."""

    def __init__(self, device: str | None = None) -> None:
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(
                f"Qwen model not found under {MODEL_DIR}. "
                "Download Qwen/Qwen3-TTS-12Hz-0.6B-Base there first."
            )
        if not REF_AUDIO.is_file():
            raise FileNotFoundError(
                f"Reference audio missing: {REF_AUDIO}. "
                "scp your HR voice wav to that path."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        self._prompt = None

        from qwen_tts import Qwen3TTSModel

        logger.info(
            "Loading model_name=%s from %s device=%s dtype=%s",
            MODEL_NAME,
            MODEL_DIR,
            self.device,
            DTYPE,
        )
        self.model = Qwen3TTSModel.from_pretrained(
            str(MODEL_DIR),
            device_map=self.device if self.device == "cpu" else "cuda:0",
            dtype=DTYPE,
        )
        logger.info(
            "Building voice-clone prompt ref=%s text_chars=%d",
            REF_AUDIO,
            len(REF_TEXT),
        )
        self._prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(REF_AUDIO),
            ref_text=REF_TEXT,
        )
        logger.info("Qwen voice-clone prompt ready (voice_name=%s)", VOICE_NAME)

    def warm(self, _voice: str | None = None) -> None:
        # Model + prompt already built in __init__.
        return

    def synthesize_array(self, text: str, language: str | None = None) -> np.ndarray:
        lang = (language or LANGUAGE_NAME).strip() or LANGUAGE_NAME
        with self.lock:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=lang,
                voice_clone_prompt=self._prompt,
            )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if int(sr) != self.sample_rate:
            logger.warning("Qwen returned sr=%s; expected %s", sr, self.sample_rate)
            self.sample_rate = int(sr)
        return audio

    def iter_audio(self, text: str, language: str | None = None):
        """Yield float32 PCM chunks (pseudo-stream after full generate)."""
        audio = self.synthesize_array(text, language=language)
        if audio.size == 0:
            return
        for start in range(0, audio.size, _CHUNK_SAMPLES):
            yield audio[start : start + _CHUNK_SAMPLES]

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        audio = self.synthesize_array(text, language=language)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
