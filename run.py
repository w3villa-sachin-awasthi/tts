#!/usr/bin/env python3
"""Launch exactly one TTS model server. Only that model's weights load.

Usage:
  python run.py kokoro
  python run.py speecht5
  python run.py chatterbox
  python run.py cosyvoice
  python run.py indic-parler

Stop the process (Ctrl+C) before starting a different model — all share port 8765.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS = {
    "kokoro": "server_kokoro.py",
    "speecht5": "server_speecht5.py",
    "chatterbox": "server_chatterbox.py",
    "cosyvoice": "server_cosyvoice.py",
    "indic-parler": "server_indic_parler.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single local TTS model (isolates RAM/GPU from other models).",
    )
    parser.add_argument(
        "model",
        choices=sorted(MODELS),
        help="Which model server to start. Only this model is imported and loaded.",
    )
    args = parser.parse_args()
    script = ROOT / MODELS[args.model]
    if not script.is_file():
        print(f"Missing server script: {script}", file=sys.stderr)
        sys.exit(1)
    print(f"Starting {args.model} only → {script.name}", flush=True)
    print("Stop this process before launching another model.", flush=True)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
