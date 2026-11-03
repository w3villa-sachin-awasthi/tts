#!/usr/bin/env python3
"""Verify Kokoro/Goonj SSE streaming against a running server."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from log_config import setup_logger

logger = setup_logger(__name__)


def read_sse_events(url: str, payload: dict) -> list[tuple[str, dict]]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[tuple[str, dict]] = []
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.headers.get("Content-Type", "").split(";")[0] != "text/event-stream":
            raise RuntimeError(f"Expected SSE, got {response.headers.get('Content-Type')}")
        buffer = ""
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                event = "message"
                data = ""
                for line in block.split("\n"):
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                events.append((event, json.loads(data) if data else {}))
    return events


def check_model(base: str, route: str, payload: dict, name: str) -> None:
    events = read_sse_events(base + route, {**payload, "stream": True})
    kinds = [e[0] for e in events]
    if "error" in kinds:
        err = next(v for k, v in events if k == "error")
        raise RuntimeError(f"{name} stream error: {err.get('message')}")
    if "meta" not in kinds or "end" not in kinds:
        raise RuntimeError(f"{name} missing meta/end: {kinds}")
    chunks = sum(1 for k, _ in events if k == "chunk")
    if chunks < 1:
        raise RuntimeError(f"{name} produced no audio chunks")
    logger.info("OK %s: %s chunk(s), events=%s", name, chunks, kinds)


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    # Active engine only responds 200; the other returns 503 — test goonj by default.
    active = json.loads(urllib.request.urlopen(base + "/api/active", timeout=30).read())
    model = active.get("active_model")
    if model == "goonj":
        check_model(
            base,
            "/api/goonj",
            {"input": "Streaming test.", "voice": "en_priya", "language": "a"},
            "Goonj",
        )
    elif model == "kokoro":
        check_model(
            base,
            "/api/speech",
            {"input": "Streaming test.", "voice": "af_heart", "language": "a"},
            "Kokoro",
        )
    else:
        raise SystemExit(f"Unexpected active_model={model!r}")
    logger.info("Streaming check passed for %s", model)


if __name__ == "__main__":
    main()
