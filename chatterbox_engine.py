"""ResembleAI Chatterbox Multilingual TTS. Loaded on first use."""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

# Mirrors chatterbox.mtl_tts.SUPPORTED_LANGUAGES
LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

VOICES = [
    ("default", "Default"),
    ("expressive", "Expressive"),
]
SAMPLE_RATE = 24000
REF_DIR = Path(__file__).resolve().parent / "voices"


class ChatterboxEngine:
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
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            try:
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            except Exception:
                if self.device == "cpu":
                    raise
                self.device = "cpu"
                self.model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
            self.sr = int(getattr(self.model, "sr", SAMPLE_RATE))
            self._loaded = True

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en",
        audio_prompt_path: str | None = None,
    ) -> bytes:
        lang = (language or "en").lower()
        if lang not in LANGUAGES:
            raise ValueError(f"Unsupported Chatterbox language: {language}")
        if voice not in {key for key, _label in VOICES} and not audio_prompt_path:
            raise ValueError("Unknown Chatterbox voice.")
        self.load()
        exaggeration = 0.7 if voice == "expressive" else 0.5
        prompt = audio_prompt_path
        if prompt is None and voice not in {"default", "expressive"}:
            candidate = REF_DIR / f"{voice}.wav"
            if candidate.is_file():
                prompt = str(candidate)
        try:
            return self._generate(
                text,
                language_id=lang,
                exaggeration=exaggeration,
                audio_prompt_path=prompt,
            )
        except RuntimeError:
            if self.device == "cpu":
                raise
            with self.lock:
                self.device = "cpu"
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                self.model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
                self.sr = int(getattr(self.model, "sr", SAMPLE_RATE))
            return self._generate(
                text,
                language_id=lang,
                exaggeration=exaggeration,
                audio_prompt_path=prompt,
            )

    def _generate(
        self,
        text: str,
        *,
        language_id: str,
        exaggeration: float,
        audio_prompt_path: str | None,
    ) -> bytes:
        with self.lock:
            kwargs: dict = {
                "language_id": language_id,
                "exaggeration": exaggeration,
            }
            if audio_prompt_path:
                kwargs["audio_prompt_path"] = audio_prompt_path
            wav = self.model.generate(text, **kwargs)
        if hasattr(wav, "detach"):
            audio = wav.squeeze().detach().float().cpu().numpy()
        else:
            audio = np.asarray(wav, dtype=np.float32).squeeze()
        if audio.size == 0:
            raise RuntimeError("Chatterbox returned no audio.")
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), self.sr, format="WAV")
        return buffer.getvalue()
