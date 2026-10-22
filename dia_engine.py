"""Nari Dia 1.6B — English dialogue TTS (single-speaker via [S1] for calls)."""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL_ID = os.environ.get("DIA_MODEL_ID", "nari-labs/Dia-1.6B").strip()
SAMPLE_RATE = 44100


def _dialogue_text(text: str) -> str:
    t = text.strip()
    if t.startswith("[S1]") or t.startswith("[S2]"):
        return t
    return f"[S1] {t}"


def _audio_prompt(voice: str) -> str | None:
    if voice == "default":
        return None
    path = Path(voice)
    if path.is_file():
        return str(path)
    raise ValueError(
        f"Unknown Dia voice {voice!r}. Use 'default' or a path to a reference .wav file."
    )


class DiaEngine:
    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self._loaded = False
        self.model = None
        self.sample_rate = SAMPLE_RATE

    def load(self) -> None:
        if self._loaded:
            return
        with self.lock:
            if self._loaded:
                return
            from dia.model import Dia

            dtype = "float16" if self.device == "cuda" else "float32"
            dev = torch.device(self.device)
            self.model = Dia.from_pretrained(MODEL_ID, compute_dtype=dtype, device=dev)
            self._loaded = True

    def synthesize_array(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        self.load()
        prompt_path = _audio_prompt(voice)
        script = _dialogue_text(text)
        kwargs: dict = {}
        if prompt_path:
            kwargs["audio_prompt"] = prompt_path
        with self.lock:
            out = self.model.generate(script, **kwargs)
        audio = np.asarray(out, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Dia returned no audio.")
        return audio, self.sample_rate

    def synthesize(self, text: str, voice: str) -> bytes:
        audio, rate = self.synthesize_array(text, voice)
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format="WAV")
        return buffer.getvalue()
