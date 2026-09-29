#!/usr/bin/env python3
"""Verify Kokoro/Piper SSE streaming against a running server."""

from __future__ import annotations

import json
import sys
import urllib.request


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
    print(f"OK {name}: {chunks} chunk(s), events={kinds}")


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    check_model(
        base,
        "/api/piper",
        {"input": "Streaming test.", "voice": "en_US-lessac-medium"},
        "Piper",
    )
    check_model(
        base,
        "/api/speech",
        {"input": "Streaming test.", "voice": "af_heart", "language": "a"},
        "Kokoro",
    )
    bad = urllib.request.Request(
        base + "/api/speecht5",
        data=json.dumps({"input": "Hi", "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(bad, timeout=30) as response:
            if response.status != 400:
                raise RuntimeError("SpeechT5 stream should return 400")
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
    print("OK SpeechT5 correctly rejects stream=true")


if __name__ == "__main__":
    main()
