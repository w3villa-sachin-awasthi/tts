"""Microsoft SpeechT5 TTS. English only, loaded on first use."""

from __future__ import annotations

import io
import threading
import zipfile

import numpy as np
import soundfile as sf
import torch

SPEAKERS = [
    ("slt", "SLT (US female)"),
    ("clb", "CLB (US female)"),
    ("bdl", "BDL (US male)"),
    ("rms", "RMS (US male)"),
    ("ksp", "KSP (Indian English)"),
    ("awb", "AWB (Scottish)"),
]
SAMPLE_RATE = 16000


class SpeechT5Engine:
    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self._loaded = False
        self.processor = None
        self.model = None
        self.vocoder = None
        self.embeddings: dict[str, torch.Tensor] = {}

    def load(self) -> None:
        if self._loaded:
            return
        with self.lock:
            if self._loaded:
                return
            from huggingface_hub import hf_hub_download
            from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            archive = hf_hub_download(
                repo_id="Matthijs/cmu-arctic-xvectors",
                filename="spkrec-xvect.zip",
                repo_type="dataset",
            )
            wanted = {key for key, _label in SPEAKERS}
            with zipfile.ZipFile(archive) as bundle:
                for name in sorted(bundle.namelist()):
                    if not name.endswith(".npy"):
                        continue
                    stem = name.rsplit("/", 1)[-1][:-4]
                    for key in wanted:
                        if key in self.embeddings:
                            continue
                        if f"cmu_us_{key}_" in stem:
                            with bundle.open(name) as handle:
                                vector = np.load(handle)
                            self.embeddings[key] = torch.tensor(vector).unsqueeze(0)
                    if len(self.embeddings) == len(wanted):
                        break
            missing = wanted - set(self.embeddings)
            if missing:
                raise RuntimeError(f"Missing SpeechT5 speakers: {', '.join(sorted(missing))}")
            self._move(self.device)
            self._loaded = True

    def _move(self, device: str) -> None:
        self.device = device
        self.model.to(device)
        self.vocoder.to(device)
        self.embeddings = {key: value.to(device) for key, value in self.embeddings.items()}

    def synthesize(self, text: str, voice: str) -> bytes:
        if voice not in {key for key, _label in SPEAKERS}:
            raise ValueError("Unknown SpeechT5 voice.")
        self.load()
        try:
            return self._generate(text, voice)
        except RuntimeError:
            if self.device == "cpu":
                raise
            with self.lock:
                self._move("cpu")
            return self._generate(text, voice)

    def _generate(self, text: str, voice: str) -> bytes:
        with self.lock:
            inputs = self.processor(text=text, return_tensors="pt")
            speech = self.model.generate_speech(
                inputs["input_ids"].to(self.device),
                self.embeddings[voice],
                vocoder=self.vocoder,
            )
        audio = speech.detach().float().cpu().numpy()
        if audio.size == 0:
            raise RuntimeError("SpeechT5 returned no audio.")
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV")
        return buffer.getvalue()
