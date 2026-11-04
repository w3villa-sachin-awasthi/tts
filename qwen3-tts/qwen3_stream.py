"""SSE helpers for Qwen3 (same float32 chunk format as Goonj)."""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.append(str(_PARENT))  # append, do not shadow local modules

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


def stream_qwen3(text: str, language: str | None, engine) -> Iterator[bytes]:
    rate = int(getattr(engine, "sample_rate", 24000) or 24000)
    yield sse_bytes(
        "meta",
        {"model": "qwen3", "sample_rate": rate, "format": "f32le"},
    )
    count = 0
    try:
        for audio in engine.iter_audio(text, language):
            count += 1
            yield sse_bytes("chunk", encode_f32_chunk(audio))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Qwen3 stream failed")
        yield sse_bytes("error", {"message": str(exc)})
        return
    if count == 0:
        yield sse_bytes("error", {"message": "Qwen3 returned no audio."})
        return
    yield sse_bytes("end", {"chunks": count, "sample_rate": rate})
