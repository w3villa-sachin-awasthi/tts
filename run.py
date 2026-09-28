#!/usr/bin/env python3
"""Launch exactly one TTS model server. Only that model's code and weights load.

Usage:
  .venv/bin/python run.py kokoro
  .venv/bin/python run.py speecht5
  .venv/bin/python run.py chatterbox
  .venv/bin/python run.py chatterbox-turbo
  .venv-cosyvoice/bin/python run.py cosyvoice
  .venv-indic-parler/bin/python run.py indic-parler

Stop with Ctrl+C before starting a different model (shared port 8765).
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
    "chatterbox-turbo": "server_chatterbox_turbo.py",
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
