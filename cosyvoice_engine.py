"""FunAudioLLM CosyVoice2-0.5B. Multilingual, loaded on first use only."""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
COSY_ROOT = ROOT / "vendor" / "CosyVoice"
MODEL_DIR = ROOT / "models" / "CosyVoice2-0.5B"
PROMPT_WAV = COSY_ROOT / "asset" / "zero_shot_prompt.wav"
# Transcript of the bundled female prompt clip
PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ru": "Russian",
}

VOICES = [
    ("default", "Default (female prompt)"),
]

SAMPLE_RATE = 24000


class CosyVoiceEngine:
    def __init__(self) -> None:
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
            if not MODEL_DIR.is_dir():
                raise RuntimeError(
                    f"CosyVoice2 model not found at {MODEL_DIR}. "
                    "Download FunAudioLLM/CosyVoice2-0.5B first."
                )
            if not PROMPT_WAV.is_file():
                raise RuntimeError(f"Missing prompt wav: {PROMPT_WAV}")

            matcha = COSY_ROOT / "third_party" / "Matcha-TTS"
            for path in (str(COSY_ROOT), str(matcha)):
                if path not in sys.path:
                    sys.path.insert(0, path)

            from cosyvoice.cli.cosyvoice import AutoModel

            # Official Mac path is CPU (no CUDA). Keeps other models unloaded.
            self.model = AutoModel(model_dir=str(MODEL_DIR))
            self.sr = int(getattr(self.model, "sample_rate", SAMPLE_RATE))
            self._loaded = True

    def unload(self) -> None:
        with self.lock:
            self.model = None
            self._loaded = False

    def synthesize(self, text: str, language: str = "en", voice: str = "default") -> bytes:
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported CosyVoice language: {language}")
        if voice not in {key for key, _ in VOICES}:
            raise ValueError("Unknown CosyVoice voice.")
        self.load()
        with self.lock:
            chunks = []
            for _i, out in enumerate(
                self.model.inference_zero_shot(
                    text,
                    PROMPT_TEXT,
                    str(PROMPT_WAV),
                    stream=False,
                )
            ):
                speech = out["tts_speech"]
                if hasattr(speech, "detach"):
                    audio = speech.squeeze().detach().float().cpu().numpy()
                else:
                    audio = np.asarray(speech, dtype=np.float32).squeeze()
                if audio.size:
                    chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("CosyVoice returned no audio.")
        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(chunks), self.sr, format="WAV")
        return buffer.getvalue()
