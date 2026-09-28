"""Piper medium voices. Each file is the small VITS model, about 15–20 million parameters."""

from __future__ import annotations

import io
import threading

import numpy as np
import soundfile as sf

VOICES = [
    ("en-US", "English (US)", "en_US-lessac-medium", "Lessac (female)"),
    ("en-US", "English (US)", "en_US-amy-medium", "Amy (female)"),
    ("en-US", "English (US)", "en_US-ryan-medium", "Ryan (male)"),
    ("en-GB", "English (UK)", "en_GB-alan-medium", "Alan (male)"),
    ("en-GB", "English (UK)", "en_GB-cori-medium", "Cori (female)"),
    ("hi-IN", "Hindi", "hi_IN-pratham-medium", "Pratham (male)"),
    ("hi-IN", "Hindi", "hi_IN-priyamvada-medium", "Priyamvada (female)"),
    ("es-ES", "Spanish", "es_ES-davefx-medium", "Davefx (male)"),
]
SAMPLE_RATE = 22050


def voice_repo_path(voice_id: str) -> str:
    speaker_and_locale, quality = voice_id.rsplit("-", 1)
    locale, speaker = speaker_and_locale.split("-", 1)
    family = locale.split("_", 1)[0]
    return f"{family}/{locale}/{speaker}/{quality}/{voice_id}"


class PiperEngine:
    def __init__(self, model_dir) -> None:
        self.model_dir = model_dir
        self.lock = threading.Lock()
        self.loaded: dict[str, object] = {}

    def _onnx_path(self, voice_id: str) -> str:
        from huggingface_hub import hf_hub_download

        relative = voice_repo_path(voice_id)
        folder = self.model_dir / voice_id
        folder.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=f"{relative}.onnx",
            revision="v1.0.0",
            local_dir=folder,
        )
        hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=f"{relative}.onnx.json",
            revision="v1.0.0",
            local_dir=folder,
        )
        return str(folder / f"{relative}.onnx")

    def voice(self, voice_id: str):
        if voice_id not in {item[2] for item in VOICES}:
            raise ValueError("Unknown Piper voice.")
        if voice_id not in self.loaded:
            from piper import PiperVoice

            self.loaded[voice_id] = PiperVoice.load(self._onnx_path(voice_id))
        return self.loaded[voice_id]

    def synthesize(self, text: str, voice_id: str) -> bytes:
        with self.lock:
            chunks = []
            sample_rate = SAMPLE_RATE
            for chunk in self.voice(voice_id).synthesize(text):
                audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
                if audio.size:
                    chunks.append(audio)
                    sample_rate = chunk.sample_rate
        if not chunks:
            raise RuntimeError("Piper returned no audio.")
        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(chunks), sample_rate, format="WAV")
        return buffer.getvalue()
