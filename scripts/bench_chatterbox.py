#!/usr/bin/env python3
"""Bench Chatterbox Multilingual V3 (full WAV — no fake streaming).

Usage (server running with MODEL_NAME=chatterbox):
  python scripts/bench_chatterbox.py
  python scripts/bench_chatterbox.py http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import resource
import sys
import time
import urllib.request
import wave
from io import BytesIO


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def synth(base: str, text: str, language: str) -> tuple[float, float, bytes]:
    body = json.dumps(
        {"input": text, "voice": "ref_clone", "language": language, "stream": False}
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/api/chatterbox",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    wall = time.perf_counter() - t0
    with wave.open(BytesIO(data), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        dur = frames / float(sr) if sr else 0.0
    return wall, dur, data


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    cases = [
        ("en", "I'm calling from W three Villa Technologies. Please note that this call will be recorded."),
        ("hi", "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?"),
        ("en", "Namaste, main aapki interview schedule confirm karne ke liye call kar raha hoon."),
    ]
    print("warmup…")
    synth(base, "Warmup one two three.", "en")
    for lang, text in cases:
        wall, dur, _ = synth(base, text, lang)
        rtf = (wall / dur) if dur > 0 else 0.0
        print(
            f"lang={lang}  wall_s={wall:.2f}  audio_s={dur:.2f}  "
            f"RTF={rtf:.2f}  (gen/audio)  peak_rss_mb≈{_rss_mb():.0f}"
        )
        print(f"  text={text[:60]}…")


if __name__ == "__main__":
    main()
