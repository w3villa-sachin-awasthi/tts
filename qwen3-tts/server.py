#!/usr/bin/env python3
"""Qwen3-TTS clone server. Parallel to /data/tts Goonj on 8765.

Settings: edit active_qwen3.py (no .env). Restart: pm2 restart tts-qwen3
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_PARENT = ROOT.parent
# Local package first; parent only for log_config (append, never shadow).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_PARENT) not in sys.path:
    sys.path.append(str(_PARENT))

from log_config import setup_logger

from active_qwen3 import (
    LANGUAGE_NAME,
    MODEL_ID,
    MODEL_NAME,
    ROUTE,
    TTS_HOST,
    TTS_PORT,
    VOICE_NAME,
)
from qwen3_engine import Qwen3Engine, normalize_language
from qwen3_stream import stream_qwen3

logger = setup_logger(__name__)

_engine: Qwen3Engine | None = None


def get_engine() -> Qwen3Engine:
    global _engine
    if _engine is None:
        _engine = Qwen3Engine()
        _engine.warm()
    return _engine


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, events) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in events:
            self.wfile.write(chunk)
            self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/health"):
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        if path == "/api/active":
            eng = get_engine()
            payload = {
                "active_model": MODEL_NAME,
                "active_voice": VOICE_NAME,
                "active_language": LANGUAGE_NAME,
                "route": ROUTE,
                "sample_rate": eng.sample_rate,
                "device": eng.device,
                "prompt_path": str(eng.prompt_path),
                "model_id": MODEL_ID,
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != ROUTE:
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode())
        except json.JSONDecodeError:
            self._send(400, b"Invalid JSON", "text/plain; charset=utf-8")
            return
        text = str(payload.get("input", "")).strip()
        if not text:
            self._send(400, b"Text is required", "text/plain; charset=utf-8")
            return
        lang = normalize_language(str(payload.get("language") or LANGUAGE_NAME))
        want_stream = bool(payload.get("stream", True))
        if not want_stream:
            self._send(
                400,
                b"Only stream=true is supported on /api/qwen3",
                "text/plain; charset=utf-8",
            )
            return
        try:
            self._send_sse(stream_qwen3(text, lang, get_engine()))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.info("SSE %s client gone", ROUTE)
        except Exception as exc:  # noqa: BLE001
            logger.exception("POST %s failed", ROUTE)
            try:
                self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass


def main() -> None:
    get_engine()
    base = (
        f"http://{TTS_HOST}:{TTS_PORT}"
        if TTS_HOST not in ("0.0.0.0", "::")
        else f"http://127.0.0.1:{TTS_PORT}"
    )
    logger.info(
        "Active hardcoded model=%s voice=%s lang=%s route=%s",
        MODEL_NAME,
        VOICE_NAME,
        LANGUAGE_NAME,
        ROUTE,
    )
    logger.info("Listening on %s:%s", TTS_HOST, TTS_PORT)
    logger.info("API: %s%s  |  status: %s/api/active", base, ROUTE, base)
    ThreadingHTTPServer((TTS_HOST, TTS_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
