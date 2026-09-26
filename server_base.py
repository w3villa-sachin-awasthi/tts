"""Shared HTTP helpers for single-model TTS servers."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable


class SingleModelHandler(BaseHTTPRequestHandler):
    """Subclass and set class attrs: page_bytes, api_models, route, synthesize_fn, wav_name, output_dir."""

    page_bytes: bytes = b""
    page_paths: tuple[str, ...] = ("/", "/index.html")
    api_models: dict[str, Any] = {}
    route: str = ""
    synthesize_fn: Callable[[dict[str, Any]], bytes] | None = None
    wav_name: str = "out.wav"
    output_dir: Path = Path("output")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in self.page_paths:
            self._send(200, self.page_bytes, "text/html; charset=utf-8")
            return
        if path == "/api/models":
            self._send(200, json.dumps(self.api_models).encode(), "application/json")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != self.route or self.synthesize_fn is None:
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
        try:
            audio = self.synthesize_fn(payload)
        except Exception as exc:  # noqa: BLE001
            self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.wav_name).write_bytes(audio)
        self._send(200, audio, "audio/wav")
