#!/usr/bin/env python3
"""Build voice_clone_prompt once from ref in active_qwen3.py → cache/."""

from __future__ import annotations

from pathlib import Path

from active_qwen3 import PROMPT_PATH, REF_PATH, REF_TEXT, X_VECTOR_ONLY
from qwen3_engine import Qwen3Engine


def main() -> None:
    ref = Path(REF_PATH)
    out = Path(PROMPT_PATH)
    if not ref.is_file():
        raise SystemExit(
            f"Reference audio not found: {ref}\n"
            "Put your 15s wav there, or edit REF_PATH in active_qwen3.py"
        )

    engine = Qwen3Engine()
    engine.build_prompt(
        ref,
        ref_text=REF_TEXT or None,
        x_vector_only=X_VECTOR_ONLY,
        save_path=out,
    )
    print(f"OK prompt saved -> {out}")
    print(f"X_VECTOR_ONLY={X_VECTOR_ONLY}")


if __name__ == "__main__":
    main()
