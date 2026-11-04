#!/usr/bin/env python3
"""Extract speaker embedding once (TRT ONNX / bundle script) → cache/speaker_emb.b64."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from active_qwen3 import (
    EMBED_PATH,
    EXTRACT_SCRIPT,
    REF_PATH,
    SPEAKER_ENCODER_ONNX,
)


def main() -> None:
    ref = Path(REF_PATH)
    out = Path(EMBED_PATH)
    onnx = Path(SPEAKER_ENCODER_ONNX)
    script = Path(EXTRACT_SCRIPT)

    if not ref.is_file():
        raise SystemExit(f"Reference audio not found: {ref}")

    out.parent.mkdir(parents=True, exist_ok=True)

    if script.is_file() and onnx.is_file():
        subprocess.check_call(
            [sys.executable, str(script), str(ref), str(onnx), str(out)],
        )
        print(f"OK embed saved -> {out}")
        return

    if not onnx.is_file():
        raise SystemExit(
            f"Missing speaker encoder ONNX: {onnx}\n"
            "After TRT download, set SPEAKER_ENCODER_ONNX in active_qwen3.py "
            "to the bundle's speaker_encoder.onnx path."
        )
    if not script.is_file():
        raise SystemExit(
            f"Missing extract script: {script}\n"
            "Copy scripts/extract_speaker_embedding.py from the TRT bundle, "
            "or set EXTRACT_SCRIPT in active_qwen3.py."
        )


if __name__ == "__main__":
    main()
