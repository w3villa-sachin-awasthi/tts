"""Indic Parler-TTS — Indian English female voices only (pinned caption)."""

from __future__ import annotations

import io
import threading

import numpy as np
import soundfile as sf
import torch

MODEL_ID = "ai4bharat/indic-parler-tts"

# English female speakers from Indic Parler-TTS (Indian English).
VOICES = (
    "Mary",
    "Meera",
    "Sneha",
    "Priya",
    "Nisha",
    "Kavya",
    "Riya",
    "Swapna",
    "Tisha",
    "Gauri",
)

_DESC = (
    "{name}'s voice is clear and slightly expressive, with a moderate speed and pitch. "
    "The recording is of very high quality, with the speaker's voice sounding clear and very close up."
)


class ParlerEngine:
    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.desc_tokenizer = None
        self.sample_rate = 44100

    def load(self) -> None:
        if self._loaded:
            return
        with self.lock:
            if self._loaded:
                return
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            self.model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(
                self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.desc_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path
            )
            self.sample_rate = int(self.model.config.sampling_rate)
            self.model.eval()
            self._loaded = True

    def synthesize_array(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        if voice not in VOICES:
            raise ValueError(f"Unknown Parler voice: {voice!r}. Use one of: {', '.join(VOICES)}")
        self.load()
        description = _DESC.format(name=voice)
        with self.lock:
            desc = self.desc_tokenizer(description, return_tensors="pt").to(self.device)
            prompt = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generation = self.model.generate(
                    input_ids=desc.input_ids,
                    attention_mask=desc.attention_mask,
                    prompt_input_ids=prompt.input_ids,
                    prompt_attention_mask=prompt.attention_mask,
                )
            audio = generation.detach().float().cpu().numpy().squeeze()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Parler returned no audio.")
        return audio, self.sample_rate

    def synthesize(self, text: str, voice: str) -> bytes:
        audio, rate = self.synthesize_array(text, voice)
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format="WAV")
        return buffer.getvalue()
