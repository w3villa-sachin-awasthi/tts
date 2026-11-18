"""Mithi v2 Flash (Oomni/mithi-v2-flash) — local 24 kHz TTS for the HTTP server.

Weights: set MITHI_MODEL_DIR, or reuse Jetson voice-assistant-test download.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from log_config import setup_logger

logger = setup_logger(__name__)

SAMPLE_RATE = 24000

MITHI_VOICES: tuple[str, ...] = (
    "satvik",
    "anirban",
    "nakul",
    "vasant",
    "siddhant",
    "charvi",
    "ahana",
    "vaidehi",
    "malini",
    "jahnavi",
)

# Mithi lang codes (not Kokoro letter codes).
MITHI_LANGS: tuple[str, ...] = (
    "en",
    "hi",
    "bn",
    "gu",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
)

_CHUNK_SAMPLES = 4800  # ~200 ms @ 24 kHz for SSE

_import_lock = threading.Lock()
_Mithi = None
_SR = SAMPLE_RATE


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def default_model_dir() -> Path:
    raw = _env("MITHI_MODEL_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    for candidate in (
        Path("/data/voice-assistant-test/models/mithi-v2-flash"),
        Path("/data/tts/models/mithi-v2-flash"),
        Path(__file__).resolve().parent / "models" / "mithi-v2-flash",
    ):
        if (candidate / "run.py").is_file() and (candidate / "config.json").is_file():
            return candidate.resolve()
    return Path("/data/voice-assistant-test/models/mithi-v2-flash")


def _resolve_device(device: str | None) -> str | None:
    """None = Mithi auto (cuda if available else cpu)."""
    if device is None:
        wanted = (_env("MITHI_DEVICE") or "auto").lower()
    else:
        wanted = device.strip().lower()
    if wanted in ("", "auto"):
        return None
    if wanted in ("cuda", "cpu"):
        return wanted
    # Map mps/other → auto so Jetson detect_device() does not break Mithi.
    return None


def _import_mithi(model_dir: Path):
    global _Mithi, _SR
    with _import_lock:
        if _Mithi is not None:
            return _Mithi, _SR

        run_path = model_dir / "run.py"
        model_str = str(model_dir)
        if model_str not in sys.path:
            sys.path.insert(0, model_str)

        spec = importlib.util.spec_from_file_location("mithi_v2_flash_run", run_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Mithi run.py from {run_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mithi_v2_flash_run"] = mod
        spec.loader.exec_module(mod)

        if not hasattr(mod, "Mithi"):
            raise ImportError(f"{run_path} did not export Mithi")
        _Mithi = mod.Mithi
        _SR = int(getattr(mod, "SR", SAMPLE_RATE))
        logger.info("Imported Mithi from %s (SR=%s)", run_path, _SR)
        return _Mithi, _SR


class MithiEngine:
    """One Mithi checkpoint in RAM (ACTIVE_TTS single-engine policy)."""

    def __init__(self, device: str | None = None) -> None:
        self.root = default_model_dir()
        if not (self.root / "run.py").is_file() or not (self.root / "config.json").is_file():
            raise FileNotFoundError(
                f"Mithi model not found under {self.root}. "
                "Set MITHI_MODEL_DIR to the mithi-v2-flash folder "
                "(needs run.py + config.json), e.g. "
                "/data/voice-assistant-test/models/mithi-v2-flash"
            )

        Mithi, sr = _import_mithi(self.root)
        resolved = _resolve_device(device)
        self._engine = Mithi(str(self.root), device=resolved)
        self.sample_rate = int(getattr(self._engine, "SR", sr) or SAMPLE_RATE)
        self.device = str(getattr(self._engine, "dev", resolved or "auto"))
        self.lock = threading.Lock()
        voices = getattr(self._engine, "voices", list(MITHI_VOICES))
        logger.info(
            "Mithi loaded root=%s device=%s voices=%s",
            self.root,
            self.device,
            ",".join(voices) if isinstance(voices, (list, tuple)) else voices,
        )

    def warm(self, voice: str, lang: str = "en") -> None:
        """Tiny warmup synth so first call latency is paid at startup."""
        with self.lock:
            _ = self._engine.synthesize("Hello.", lang=lang, voice=voice)

    def _synth_f32(self, text: str, voice: str, lang: str) -> np.ndarray:
        cleaned = text.strip()
        if not cleaned:
            return np.zeros(0, dtype=np.float32)
        if voice not in MITHI_VOICES:
            raise ValueError(
                f"Unknown Mithi voice {voice!r}. Use one of: {', '.join(MITHI_VOICES)}"
            )
        if lang not in MITHI_LANGS:
            raise ValueError(
                f"Unknown Mithi language {lang!r}. Use one of: {', '.join(MITHI_LANGS)}"
            )
        with self.lock:
            audio = self._engine.synthesize(cleaned, lang=lang, voice=voice)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def iter_audio(self, text: str, voice: str, lang: str = "en"):
        """Full synthesize then yield float32 PCM chunks (pseudo-stream for SSE)."""
        wave = self._synth_f32(text, voice, lang)
        if wave.size == 0:
            return
        for start in range(0, wave.size, _CHUNK_SAMPLES):
            yield wave[start : start + _CHUNK_SAMPLES]

    def synthesize(self, text: str, voice: str, lang: str = "en") -> bytes:
        wave = self._synth_f32(text, voice, lang)
        buf = io.BytesIO()
        sf.write(buf, wave, self.sample_rate, format="WAV")
        return buf.getvalue()
