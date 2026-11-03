"""Qwen3-TTS-12Hz-0.6B-Base voice-clone via faster-qwen3-tts (Jetson).

Caches speaker embedding once; streams audio chunks during generation.
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
SPEAKER_PT = Path("/data/qwen3-tts/ref/hr_speaker.pt")
DTYPE = torch.float32  # float16 often NaNs on Jetson Orin
XVEC_ONLY = True
STREAM_CHUNK_SIZE = 4  # ~333ms/chunk; raise to 8 if RTF < 1 on NX

# Used only when XVEC_ONLY is False (full ICL). Exact transcript of REF_AUDIO.
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


class QwenEngine:
    """One FasterQwen checkpoint + one cached speaker prompt."""

    def __init__(self, device: str | None = None) -> None:
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(
                f"Qwen model not found under {MODEL_DIR}. "
                "Download Qwen/Qwen3-TTS-12Hz-0.6B-Base there first."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()

        from faster_qwen3_tts import FasterQwen3TTS

        logger.info(
            "Loading model_name=%s from %s device=%s dtype=%s xvec_only=%s",
            MODEL_NAME,
            MODEL_DIR,
            self.device,
            DTYPE,
            XVEC_ONLY,
        )
        self.model = FasterQwen3TTS.from_pretrained(
            str(MODEL_DIR),
            device="cuda" if self.device.startswith("cuda") else self.device,
            dtype=DTYPE,
        )
        self._prompt = self._load_or_build_prompt()
        try:
            self.model.warmup(prefill_len=100)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qwen warmup skipped: %s", exc)
        logger.info("Qwen voice-clone ready (voice_name=%s)", VOICE_NAME)

    def _load_or_build_prompt(self):
        if XVEC_ONLY and SPEAKER_PT.is_file():
            logger.info("Loading cached speaker embedding %s", SPEAKER_PT)
            spk = torch.load(SPEAKER_PT, map_location="cpu", weights_only=True)
            return {"ref_spk_embedding": [spk.to(self.device)]}

        if not REF_AUDIO.is_file():
            raise FileNotFoundError(
                f"Reference audio missing: {REF_AUDIO}. "
                f"scp HR voice wav there, or provide {SPEAKER_PT}."
            )

        logger.info(
            "Extracting voice prompt ref=%s xvec_only=%s",
            REF_AUDIO,
            XVEC_ONLY,
        )
        items = self.model.model.create_voice_clone_prompt(
            ref_audio=str(REF_AUDIO),
            ref_text="" if XVEC_ONLY else REF_TEXT,
            x_vector_only_mode=XVEC_ONLY,
        )
        if XVEC_ONLY:
            spk = items[0].ref_spk_embedding.detach().cpu()
            SPEAKER_PT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(spk, SPEAKER_PT)
            logger.info("Saved speaker embedding %s", SPEAKER_PT)
            return {"ref_spk_embedding": [spk.to(self.device)]}
        return items

    def warm(self, _voice: str | None = None) -> None:
        return

    def synthesize_array(self, text: str, language: str | None = None) -> np.ndarray:
        lang = (language or LANGUAGE_NAME).strip() or LANGUAGE_NAME
        with self.lock:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=lang,
                voice_clone_prompt=self._prompt,
                xvec_only=XVEC_ONLY,
            )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if int(sr) != self.sample_rate:
            logger.warning("Qwen returned sr=%s; expected %s", sr, self.sample_rate)
            self.sample_rate = int(sr)
        return audio

    def iter_audio(self, text: str, language: str | None = None):
        """Yield float32 PCM chunks as they are generated."""
        lang = (language or LANGUAGE_NAME).strip() or LANGUAGE_NAME
        with self.lock:
            for audio_chunk, sr, _timing in self.model.generate_voice_clone_streaming(
                text=text,
                language=lang,
                voice_clone_prompt=self._prompt,
                xvec_only=XVEC_ONLY,
                chunk_size=STREAM_CHUNK_SIZE,
            ):
                if int(sr) != self.sample_rate:
                    self.sample_rate = int(sr)
                chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
                if chunk.size:
                    yield chunk

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        audio = self.synthesize_array(text, language=language)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
