"""Server-Sent Events streaming for Kokoro, Goonj, and Qwen (float32 PCM chunks)."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator

import numpy as np

from log_config import setup_logger

logger = setup_logger(__name__)


def sse_bytes(event: str, data: dict) -> bytes:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode()


def encode_f32_chunk(audio: np.ndarray) -> dict:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    return {
        "b64": base64.b64encode(samples.tobytes()).decode("ascii"),
        "samples": int(samples.size),
    }


def stream_kokoro(
    text: str,
    voice: str,
    lang: str,
    pipeline_for,
    pipeline_lock,
    voice_lang: dict,
    speed: float = 1.0,
    split_pattern: str = r"\n+",
) -> Iterator[bytes]:
    if voice not in voice_lang:
        yield sse_bytes("error", {"message": "Unknown Kokoro voice."})
        return
    yield sse_bytes("meta", {"model": "kokoro", "sample_rate": 24000, "format": "f32le"})
    count = 0
    with pipeline_lock:
        for _graphemes, _phonemes, audio in pipeline_for(lang)(
            text, voice=voice, speed=speed, split_pattern=split_pattern
        ):
            if audio is None or not len(audio):
                continue
            chunk = np.asarray(audio, dtype=np.float32)
            count += 1
            yield sse_bytes("chunk", encode_f32_chunk(chunk))
    if count == 0:
        yield sse_bytes("error", {"message": "Kokoro returned no audio."})
        return
    yield sse_bytes("end", {"chunks": count})


def stream_goonj(text: str, voice: str, goonj, speed: float = 1.0) -> Iterator[bytes]:
    """True streaming: KPipeline yields audio chunks as they are generated."""
    rate = int(getattr(goonj, "sample_rate", 24000) or 24000)
    yield sse_bytes("meta", {"model": "goonj", "sample_rate": rate, "format": "f32le"})
    count = 0
    try:
        for audio in goonj.iter_audio(text, voice, speed=speed):
            count += 1
            yield sse_bytes("chunk", encode_f32_chunk(audio))
    except Exception as exc:  # noqa: BLE001
        yield sse_bytes("error", {"message": str(exc)})
        return
    if count == 0:
        yield sse_bytes("error", {"message": "Goonj returned no audio."})
        return
    yield sse_bytes("end", {"chunks": count, "sample_rate": rate})


def stream_qwen(text: str, qwen, language: str = "English") -> Iterator[bytes]:
    """Pseudo-stream: full voice-clone generate, then emit ~20 ms f32le chunks."""
    rate = int(getattr(qwen, "sample_rate", 24000) or 24000)
    yield sse_bytes(
        "meta",
        {"model": "qwen", "sample_rate": rate, "format": "f32le", "voice": "hr_clone"},
    )
    count = 0
    try:
        for audio in qwen.iter_audio(text, language=language):
            count += 1
            yield sse_bytes("chunk", encode_f32_chunk(audio))
    except Exception as exc:  # noqa: BLE001
        yield sse_bytes("error", {"message": str(exc)})
        return
    if count == 0:
        yield sse_bytes("error", {"message": "Qwen returned no audio."})
        return
    yield sse_bytes("end", {"chunks": count, "sample_rate": rate})
