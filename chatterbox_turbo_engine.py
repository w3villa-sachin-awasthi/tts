"""ResembleAI Chatterbox-Turbo (350M, English). Loaded on first use only."""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

# Turbo is English-only (ResembleAI/chatterbox-turbo).
LANGUAGES = {
    "en": "English",
}

VOICES = [
    ("default", "Default (builtin or voices/turbo_default.wav)"),
]

SAMPLE_RATE = 24000
REF_DIR = Path(__file__).resolve().parent / "voices"
DEFAULT_REF = REF_DIR / "turbo_default.wav"


class ChatterboxTurboEngine:
    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self._loaded = False
        self.model = None
        self.sr = SAMPLE_RATE

    def load(self) -> None:
        if self._loaded:
            return
        with self.lock:
            if self._loaded:
                return
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            try:
                self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            except Exception:
                if self.device == "cpu":
                    raise
                self.device = "cpu"
                self.model = ChatterboxTurboTTS.from_pretrained(device="cpu")
            self.sr = int(getattr(self.model, "sr", SAMPLE_RATE))
            self._loaded = True

    def _resolve_prompt(self, voice: str, audio_prompt_path: str | None) -> str | None:
        if audio_prompt_path:
            return audio_prompt_path
        if voice not in {"default"}:
            candidate = REF_DIR / f"{voice}.wav"
            if candidate.is_file():
                return str(candidate)
            raise ValueError(f"Unknown Chatterbox-Turbo voice: {voice}")
        if DEFAULT_REF.is_file():
            return str(DEFAULT_REF)
        # Builtin conds.pt from the HF snapshot (if present after load).
        if self.model is not None and getattr(self.model, "conds", None) is not None:
            return None
        raise RuntimeError(
            "Chatterbox-Turbo needs a reference voice. Place a >5s WAV at "
            f"{DEFAULT_REF} or pass audio_prompt_path."
        )

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        audio_prompt_path: str | None = None,
    ) -> bytes:
        lang = (language or "en").lower()
        if lang not in LANGUAGES:
            raise ValueError("Chatterbox-Turbo is English-only (language=en).")
        if voice not in {key for key, _label in VOICES} and not audio_prompt_path:
            candidate = REF_DIR / f"{voice}.wav"
            if not candidate.is_file():
                raise ValueError("Unknown Chatterbox-Turbo voice.")
        self.load()
        prompt = self._resolve_prompt(voice, audio_prompt_path)
        try:
            return self._generate(text, audio_prompt_path=prompt)
        except RuntimeError:
            if self.device == "cpu":
                raise
            with self.lock:
                self.device = "cpu"
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                self.model = ChatterboxTurboTTS.from_pretrained(device="cpu")
                self.sr = int(getattr(self.model, "sr", SAMPLE_RATE))
            prompt = self._resolve_prompt(voice, audio_prompt_path)
            return self._generate(text, audio_prompt_path=prompt)

    def _generate(self, text: str, *, audio_prompt_path: str | None) -> bytes:
        with self.lock:
            kwargs: dict = {}
            if audio_prompt_path:
                kwargs["audio_prompt_path"] = audio_prompt_path
            wav = self.model.generate(text, **kwargs)
        if hasattr(wav, "detach"):
            audio = wav.squeeze().detach().float().cpu().numpy()
        else:
            audio = np.asarray(wav, dtype=np.float32).squeeze()
        if audio.size == 0:
            raise RuntimeError("Chatterbox-Turbo returned no audio.")
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), self.sr, format="WAV")
        return buffer.getvalue()
