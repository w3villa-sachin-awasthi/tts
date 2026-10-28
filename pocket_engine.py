"""Pocket TTS (Kyutai) — true streaming PCM via generate_audio_stream."""

from __future__ import annotations

import io
import os
import threading

import numpy as np
import soundfile as sf

from log_config import setup_logger

logger = setup_logger(__name__)

SAMPLE_RATE = 24000


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


class PocketEngine:
    """One model + one voice state in RAM (matches ACTIVE_TTS single-engine policy)."""

    def __init__(self) -> None:
        from pocket_tts import TTSModel

        config = _env("POCKET_CONFIG")
        self.model = TTSModel.load_model(config) if config else TTSModel.load_model()
        self.sample_rate = int(getattr(self.model, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        self.lock = threading.Lock()
        self._voice: str | None = None
        self._state = None
        logger.info(
            "Pocket TTS loaded config=%r sample_rate=%s",
            config or "(default english)",
            self.sample_rate,
        )

    def _state_for(self, voice: str):
        if self._voice == voice and self._state is not None:
            return self._state
        self._state = self.model.get_state_for_audio_prompt(voice)
        self._voice = voice
        return self._state

    def warm(self, voice: str) -> None:
        with self.lock:
            self._state_for(voice)

    def iter_audio(self, text: str, voice: str):
        with self.lock:
            state = self._state_for(voice)
            for chunk in self.model.generate_audio_stream(state, text):
                audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if audio.size:
                    yield audio

    def synthesize(self, text: str, voice: str) -> bytes:
        parts = list(self.iter_audio(text, voice))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
