#!/usr/bin/env python3
"""Deprecated combined entrypoint.

On branch model-isolation, each model runs in its own process so only one
loads into memory. Use:

  python run.py kokoro
  python run.py speecht5
  python run.py chatterbox
  python run.py cosyvoice
  python run.py indic-parler

Or start a server directly, e.g. python server_kokoro.py
"""

from __future__ import annotations

import sys

from run import MODELS


def main() -> None:
    names = ", ".join(sorted(MODELS))
    print(
        "Combined multi-model server.py is retired on model-isolation.\n"
        f"Start one model at a time:\n"
        f"  python run.py <model>\n"
        f"Models: {names}\n",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
