"""Qwen3-TTS-12Hz-0.6B-Base voice-clone via faster-qwen3-tts (Jetson).

Caches speaker embedding once; streams or full-emits audio under a single lock.
"""

from __future__ import annotations

import io
import threading
import time
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

# Precision: "bf16" tried first when supported; always falls back to fp32.
DTYPE_PREF = "fp32"  # set "bf16" to A/B on device; FP32 remains default-safe on Orin
XVEC_ONLY = True
# Jetson tables: larger chunk → better RTF, higher TTFA. A/B: 8 vs 12.
STREAM_CHUNK_SIZE = 12
# "full" = generate whole utterance then emit (smooth on RTF<1).
# "stream" = yield during generation (lower TTFA, underruns if RTF<1).
EMIT_MODE = "full"

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

_FRAME_SAMPLES = int(SAMPLE_RATE * 0.020)


def _bf16_supported() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        fn = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(fn):
            return bool(fn())
        major, _minor = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:  # noqa: BLE001
        return False


def _resolve_dtype() -> torch.dtype:
    pref = (DTYPE_PREF or "fp32").strip().lower()
    if pref in ("bf16", "bfloat16"):
        if _bf16_supported():
            logger.info("Qwen dtype pref=bf16 (CUDA BF16 supported)")
            return torch.bfloat16
        logger.warning("Qwen dtype pref=bf16 but BF16 unsupported; using fp32")
        return torch.float32
    if pref in ("fp16", "float16", "half"):
        logger.warning("Qwen fp16 often NaNs on Orin; prefer bf16 or fp32")
        return torch.float16
    return torch.float32


class QwenEngine:
    """One FasterQwen checkpoint + one cached speaker prompt (single-flight lock)."""

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
        self.dtype = _resolve_dtype()

        from faster_qwen3_tts import FasterQwen3TTS

        self.model = self._load_model(FasterQwen3TTS)
        self._prompt = self._load_or_build_prompt()
        self._run_warmup()
        logger.info(
            "Qwen voice-clone ready voice=%s dtype=%s emit_mode=%s chunk_size=%s",
            VOICE_NAME,
            self.dtype,
            EMIT_MODE,
            STREAM_CHUNK_SIZE,
        )

    def _load_model(self, FasterQwen3TTS):
        device_arg = "cuda" if self.device.startswith("cuda") else self.device
        try:
            logger.info(
                "Loading model_name=%s from %s device=%s dtype=%s xvec_only=%s",
                MODEL_NAME,
                MODEL_DIR,
                self.device,
                self.dtype,
                XVEC_ONLY,
            )
            return FasterQwen3TTS.from_pretrained(
                str(MODEL_DIR),
                device=device_arg,
                dtype=self.dtype,
            )
        except Exception as exc:  # noqa: BLE001
            if self.dtype == torch.float32:
                raise
            logger.warning(
                "Qwen load failed dtype=%s (%s); falling back to fp32",
                self.dtype,
                exc,
            )
            self.dtype = torch.float32
            return FasterQwen3TTS.from_pretrained(
                str(MODEL_DIR),
                device=device_arg,
                dtype=self.dtype,
            )

    def _run_warmup(self) -> None:
        t0 = time.perf_counter()
        try:
            self.model.warmup(prefill_len=100)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "Qwen CUDA-graph warmup OK prefill_len=100 elapsed_ms=%.0f",
                ms,
            )
        except Exception as exc:  # noqa: BLE001
            ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "Qwen CUDA-graph warmup FAILED after %.0fms: %s "
                "(first requests will be slower; graphs may capture lazily)",
                ms,
                exc,
            )

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
        t0 = time.perf_counter()
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
        wall = time.perf_counter() - t0
        audio_s = audio.size / float(self.sample_rate) if audio.size else 0.0
        logger.info(
            "Qwen full-gen chars=%d audio_s=%.2f wall_s=%.2f rtf=%.2f",
            len(text),
            audio_s,
            wall,
            (audio_s / wall) if wall > 0 else 0.0,
        )
        return audio

    def iter_audio(self, text: str, language: str | None = None):
        """Yield float32 PCM. EMIT_MODE=full avoids underrun when RTF<1."""
        if EMIT_MODE == "stream":
            yield from self._iter_audio_stream(text, language)
            return
        audio = self.synthesize_array(text, language=language)
        if audio.size == 0:
            return
        t_emit0 = time.perf_counter()
        n = 0
        for start in range(0, audio.size, _FRAME_SAMPLES):
            n += 1
            yield audio[start : start + _FRAME_SAMPLES]
        logger.info(
            "Qwen full-emit frames=%d emit_wall_ms=%.0f (playback-continuous)",
            n,
            (time.perf_counter() - t_emit0) * 1000,
        )

    def _iter_audio_stream(self, text: str, language: str | None = None):
        lang = (language or LANGUAGE_NAME).strip() or LANGUAGE_NAME
        t0 = time.perf_counter()
        t_first: float | None = None
        last_t: float | None = None
        gap_sum = 0.0
        gap_n = 0
        total_samples = 0
        n_chunks = 0
        with self.lock:
            for audio_chunk, sr, _timing in self.model.generate_voice_clone_streaming(
                text=text,
                language=lang,
                voice_clone_prompt=self._prompt,
                xvec_only=XVEC_ONLY,
                chunk_size=STREAM_CHUNK_SIZE,
            ):
                now = time.perf_counter()
                if int(sr) != self.sample_rate:
                    self.sample_rate = int(sr)
                chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
                if not chunk.size:
                    continue
                if t_first is None:
                    t_first = now
                    logger.info(
                        "Qwen TTFA_ms=%.0f chars=%d chunk_samples=%d chunk_size=%d",
                        (t_first - t0) * 1000,
                        len(text),
                        chunk.size,
                        STREAM_CHUNK_SIZE,
                    )
                elif last_t is not None:
                    gap_sum += (now - last_t) * 1000
                    gap_n += 1
                last_t = now
                total_samples += chunk.size
                n_chunks += 1
                yield chunk
        wall = time.perf_counter() - t0
        audio_s = total_samples / float(self.sample_rate) if total_samples else 0.0
        avg_gap = (gap_sum / gap_n) if gap_n else 0.0
        avg_chunk_ms = (1000.0 * audio_s / n_chunks) if n_chunks else 0.0
        logger.info(
            "Qwen stream-done chunks=%d audio_s=%.2f wall_s=%.2f rtf=%.2f "
            "avg_chunk_ms=%.0f avg_inter_chunk_gap_ms=%.0f",
            n_chunks,
            audio_s,
            wall,
            (audio_s / wall) if wall > 0 else 0.0,
            avg_chunk_ms,
            avg_gap,
        )

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        audio = self.synthesize_array(text, language=language)
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
