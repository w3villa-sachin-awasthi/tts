"""AI4Bharat Indic Parler-TTS. Indic languages, loaded on first use only."""

from __future__ import annotations

import io
import threading

import numpy as np
import soundfile as sf
import torch

# Official speakers (recommended first). Language is detected from the text.
SPEAKERS = {
    "hi": {
        "name": "Hindi",
        "voices": [
            ("Divya", "Divya (female, recommended)"),
            ("Rani", "Rani (female)"),
            ("Rohit", "Rohit (male, recommended)"),
            ("Aman", "Aman (male)"),
        ],
    },
    "en": {
        "name": "English",
        "voices": [
            ("Mary", "Mary (female, recommended)"),
            ("Sneha", "Sneha (female)"),
            ("Priya", "Priya (female)"),
            ("Thoma", "Thoma (male, recommended)"),
        ],
    },
    "bn": {
        "name": "Bengali",
        "voices": [
            ("Aditi", "Aditi (female, recommended)"),
            ("Rashmi", "Rashmi (female)"),
            ("Riya", "Riya (female)"),
            ("Arjun", "Arjun (male, recommended)"),
        ],
    },
    "mr": {
        "name": "Marathi",
        "voices": [
            ("Sunita", "Sunita (female, recommended)"),
            ("Radha", "Radha (female)"),
            ("Isha", "Isha (female)"),
            ("Sanjay", "Sanjay (male, recommended)"),
        ],
    },
    "gu": {
        "name": "Gujarati",
        "voices": [
            ("Neha", "Neha (female, recommended)"),
            ("Yash", "Yash (male, recommended)"),
        ],
    },
    "pa": {
        "name": "Punjabi",
        "voices": [
            ("Gurpreet", "Gurpreet (female, recommended)"),
            ("Divjot", "Divjot (male, recommended)"),
        ],
    },
    "ta": {
        "name": "Tamil",
        "voices": [
            ("Jaya", "Jaya (female, recommended)"),
            ("Kavitha", "Kavitha (female)"),
        ],
    },
    "te": {
        "name": "Telugu",
        "voices": [
            ("Lalitha", "Lalitha (female, recommended)"),
            ("Prakash", "Prakash (male, recommended)"),
            ("Kiran", "Kiran (male)"),
        ],
    },
    "kn": {
        "name": "Kannada",
        "voices": [
            ("Anu", "Anu (female, recommended)"),
            ("Vidya", "Vidya (female)"),
            ("Suresh", "Suresh (male, recommended)"),
        ],
    },
    "ml": {
        "name": "Malayalam",
        "voices": [
            ("Anjali", "Anjali (female, recommended)"),
            ("Anju", "Anju (female)"),
            ("Harish", "Harish (male, recommended)"),
        ],
    },
    "or": {
        "name": "Odia",
        "voices": [
            ("Debjani", "Debjani (female, recommended)"),
            ("Manas", "Manas (male, recommended)"),
        ],
    },
    "as": {
        "name": "Assamese",
        "voices": [
            ("Sita", "Sita (female, recommended)"),
            ("Poonam", "Poonam (female)"),
            ("Amit", "Amit (male, recommended)"),
        ],
    },
    "ne": {
        "name": "Nepali",
        "voices": [("Amrita", "Amrita (female, recommended)")],
    },
}

MODEL_ID = "ai4bharat/indic-parler-tts"
SAMPLE_RATE = 24000


def description_for(speaker: str) -> str:
    return (
        f"{speaker}'s voice is clear and expressive, with a moderate pace and pitch. "
        "The recording is very clear audio with no background noise."
    )


class IndicParlerEngine:
    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.lock = threading.Lock()
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.sr = SAMPLE_RATE

    def load(self) -> None:
        if self._loaded:
            return
        with self.lock:
            if self._loaded:
                return
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            try:
                self.model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(self.device)
            except Exception as exc:
                msg = str(exc)
                if "gated" in msg.lower() or "401" in msg or "authenticated" in msg.lower():
                    raise RuntimeError(
                        "Indic Parler-TTS is a gated Hugging Face model. "
                        "1) Open https://huggingface.co/ai4bharat/indic-parler-tts and accept access. "
                        "2) Run: huggingface-cli login   (inside .venv-indic-parler). "
                        "Then restart this server."
                    ) from exc
                if self.device == "cpu":
                    raise
                self.device = "cpu"
                self.model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to("cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path
            )
            self.sr = int(getattr(self.model.config, "sampling_rate", SAMPLE_RATE))
            self._loaded = True

    def unload(self) -> None:
        with self.lock:
            self.model = None
            self.tokenizer = None
            self.description_tokenizer = None
            self._loaded = False

    def synthesize(self, text: str, voice: str = "Divya", language: str = "hi") -> bytes:
        lang = (language or "hi").lower()
        if lang not in SPEAKERS:
            raise ValueError(f"Unsupported language: {language}")
        allowed = {key for key, _label in SPEAKERS[lang]["voices"]}
        if voice not in allowed:
            raise ValueError(f"Unknown voice '{voice}' for language '{lang}'.")
        self.load()
        try:
            return self._generate(text, voice)
        except RuntimeError:
            if self.device == "cpu":
                raise
            with self.lock:
                self.device = "cpu"
                self.model = self.model.to("cpu")
            return self._generate(text, voice)

    def _generate(self, text: str, voice: str) -> bytes:
        description = description_for(voice)
        with self.lock:
            desc_inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)
            prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            generation = self.model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
        audio = generation.detach().float().cpu().numpy().squeeze()
        if audio.size == 0:
            raise RuntimeError("Indic Parler-TTS returned no audio.")
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), self.sr, format="WAV")
        return buffer.getvalue()
