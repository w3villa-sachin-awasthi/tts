#!/usr/bin/env python3
"""Repeatable SSE bench for Qwen on Jetson (TTFA / RTF / chunk gaps).

Usage:
  python scripts/bench_qwen_sse.py
  python scripts/bench_qwen_sse.py http://127.0.0.1:8765
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request

import numpy as np


def bench(base: str, text: str) -> None:
    body = json.dumps(
        {
            "input": text,
            "voice": "hr_clone",
            "language": "English",
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/api/qwen",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    last = None
    gap_sum = 0.0
    gap_n = 0
    n_samp = 0
    n_chunks = 0
    sr = 24000
    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = ""
        event = None
        while True:
            raw = resp.read(4096)
            if not raw:
                break
            buf += raw.decode("utf-8", "replace")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                for line in block.split("\n"):
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:") and event:
                        data = json.loads(line[5:].strip() or "{}")
                        if event == "meta":
                            sr = int(data.get("sample_rate") or sr)
                        if event == "chunk" and data.get("b64"):
                            now = time.perf_counter()
                            if first is None:
                                first = now
                            elif last is not None:
                                gap_sum += (now - last) * 1000
                                gap_n += 1
                            last = now
                            n_samp += len(
                                np.frombuffer(base64.b64decode(data["b64"]), np.float32)
                            )
                            n_chunks += 1
                        event = None
    wall = time.perf_counter() - t0
    audio_s = n_samp / float(sr) if sr else 0.0
    print(
        f"TTFA_ms={(first - t0) * 1000:.0f}  audio_s={audio_s:.2f}  "
        f"wall_s={wall:.2f}  RTF={audio_s / wall:.2f}  "
        f"chunks={n_chunks}  avg_chunk_ms={(1000 * audio_s / n_chunks) if n_chunks else 0:.0f}  "
        f"avg_gap_ms={(gap_sum / gap_n) if gap_n else 0:.0f}"
    )


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    text = (
        "I'm calling from W three Villa Technologies. "
        "Please note that this call will be recorded."
    )
    print("Note: A/B STREAM_CHUNK_SIZE=4|8|12 and DTYPE_PREF in qwen_engine.py, then restart tts.")
    print("warmup…")
    bench(base, "Warmup one two three.")
    print("measure…")
    bench(base, text)


if __name__ == "__main__":
    main()
