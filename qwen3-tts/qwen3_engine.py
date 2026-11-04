"""Qwen3-TTS Base voice clone — prompt built once, cached, reused."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

# Reuse parent TTS logger (log_config.py lives in /data/tts).
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.append(str(_PARENT))  # append — do not shadow local modules


from log_config import setup_logger

from active_qwen3 import (
    LANGUAGE_NAME,
    MODEL_ID,
    PROMPT_PATH,
    REF_PATH,
    REF_TEXT,
    SAMPLE_RATE,
    X_VECTOR_ONLY,
)

logger = setup_logger(__name__)

# Yield size when model has no native stream API (~200 ms @ 24 kHz).
FALLBACK_CHUNK = 4800

LANG_MAP = {
    "a": "English",
    "b": "English",
    "en": "English",
    "english": "English",
    "h": "Hindi",
    "hi": "Hindi",
    "hindi": "Hindi",
    "auto": "Auto",
}


def default_model_id() -> str:
    return MODEL_ID


def default_prompt_path() -> Path:
    return Path(PROMPT_PATH)


def default_ref_path() -> Path:
    return Path(REF_PATH)


def normalize_language(lang: str | None) -> str:
    if not lang:
        return LANGUAGE_NAME
    key = lang.strip().lower()
    return LANG_MAP.get(key, lang.strip())


class Qwen3Engine:
    """One Base checkpoint + one cached voice_clone_prompt in RAM."""

    def __init__(self, device: str | None = None) -> None:
        from qwen_tts import Qwen3TTSModel

        self.model_id = default_model_id()
        self.prompt_path = default_prompt_path()
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        self._prompt = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        kwargs: dict = {
            "device_map": self.device if self.device == "cpu" else "cuda:0",
            "dtype": dtype,
        }
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

        logger.info("Loading Qwen3-TTS model_id=%s device=%s", self.model_id, self.device)
        self.model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
        self.sample_rate = int(
            getattr(self.model, "sample_rate", None)
            or getattr(getattr(self.model, "model", None), "sample_rate", SAMPLE_RATE)
            or SAMPLE_RATE
        )
        logger.info("Qwen3-TTS loaded sample_rate=%s", self.sample_rate)

    def build_prompt(
        self,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        *,
        x_vector_only: bool | None = None,
        save_path: Path | None = None,
    ):
        """Extract clone features once from reference audio and save."""
        ref_audio = str(ref_audio or default_ref_path())
        if x_vector_only is None:
            x_vector_only = X_VECTOR_ONLY
        if ref_text is None:
            ref_text = REF_TEXT

        kwargs: dict = {
            "ref_audio": ref_audio,
            "x_vector_only_mode": x_vector_only,
        }
        if not x_vector_only:
            if not (ref_text or "").strip():
                raise ValueError(
                    "REF_TEXT required when X_VECTOR_ONLY=False (edit active_qwen3.py)"
                )
            kwargs["ref_text"] = ref_text.strip()

        items = self.model.create_voice_clone_prompt(**kwargs)
        out = Path(save_path or self.prompt_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(items, out)
        self._prompt = items
        self.prompt_path = out
        logger.info(
            "Voice clone prompt saved path=%s x_vector_only=%s",
            out,
            x_vector_only,
        )
        return items

    def load_prompt(self, path: Path | None = None):
        path = Path(path or self.prompt_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing voice clone prompt: {path}. "
                "Run: python extract_embed.py"
            )
        self._prompt = torch.load(path, map_location="cpu", weights_only=False)
        self.prompt_path = path
        logger.info("Voice clone prompt loaded path=%s", path)
        return self._prompt

    def ensure_prompt(self):
        if self._prompt is not None:
            return self._prompt
        return self.load_prompt()

    def warm(self) -> None:
        with self.lock:
            self.ensure_prompt()

    def iter_audio(self, text: str, language: str | None = None) -> Iterator[np.ndarray]:
        text = (text or "").strip()
        if not text:
            return
        lang = normalize_language(language or LANGUAGE_NAME)
        with self.lock:
            prompt = self.ensure_prompt()
            stream_fn = getattr(self.model, "stream_generate_voice_clone", None)
            if callable(stream_fn):
                for chunk, sr in stream_fn(
                    text=text,
                    language=lang,
                    voice_clone_prompt=prompt,
                ):
                    if sr:
                        self.sample_rate = int(sr)
                    audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    if audio.size:
                        yield audio
                return

            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=lang,
                voice_clone_prompt=prompt,
            )
            if sr:
                self.sample_rate = int(sr)
            wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            for i in range(0, wav.size, FALLBACK_CHUNK):
                piece = wav[i : i + FALLBACK_CHUNK]
                if piece.size:
                    yield piece
