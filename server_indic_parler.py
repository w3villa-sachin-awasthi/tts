#!/usr/bin/env python3
"""Indic Parler-TTS only. Does not import or load CosyVoice/Kokoro/Chatterbox."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "huggingface"
OUTPUT_DIR = ROOT / "output"
UI_PORT = 8765

os.environ.setdefault("HF_HOME", str(MODEL_DIR))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from indic_parler_engine import SPEAKERS, IndicParlerEngine

engine = IndicParlerEngine()

LANG_CATALOG = [
    {
        "code": code,
        "name": meta["name"],
        "voices": [{"id": key, "label": label} for key, label in meta["voices"]],
    }
    for code, meta in SPEAKERS.items()
]

MODELS = [
    {
        "id": "indic-parler",
        "name": "Indic Parler-TTS",
        "route": "/api/indic-parler",
        "note": "Indic Parler only — other models stay unloaded. Hindi female: Divya / Rani.",
        "languages": LANG_CATALOG,
    },
]

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Indic Parler-TTS</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f1ea;
      --ink: #1c1915;
      --muted: #6b645c;
      --card: #fffdf8;
      --line: #e4ddd2;
      --accent: #0f6e56;
      --accent-ink: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Iowan Old Style", Palatino, "Palatino Linotype", Georgia, serif;
      background: radial-gradient(1200px 500px at 10% -10%, #efe7d6, transparent), var(--bg);
      color: var(--ink);
    }
    main { max-width: 720px; margin: 0 auto; padding: 48px 20px 64px; }
    h1 { font-size: 2.4rem; font-weight: 600; margin: 0 0 8px; }
    p.lead { margin: 0 0 28px; color: var(--muted); font-family: ui-sans-serif, system-ui, sans-serif; }
    form, .result {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
    }
    label { display: block; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.82rem; margin-bottom: 6px; color: var(--muted); }
    textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      font: 1rem/1.45 ui-sans-serif, system-ui, sans-serif;
      padding: 12px;
    }
    textarea { min-height: 140px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
    button {
      margin-top: 16px;
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: var(--accent-ink);
      font: 600 0.95rem/1 ui-sans-serif, system-ui, sans-serif;
      padding: 12px 18px;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: progress; }
    .result { margin-top: 16px; display: none; }
    .result.show { display: block; }
    audio { width: 100%; margin-top: 8px; }
    .status { margin-top: 10px; font-family: ui-sans-serif, system-ui, sans-serif; color: var(--muted); min-height: 1.2em; }
    .error { color: #9b2c2c; }
  </style>
</head>
<body>
  <main>
    <h1 id="title">Indic Parler-TTS</h1>
    <p class="lead" id="lead">Indic Parler only — other models stay unloaded.</p>
    <form id="form">
      <label for="text">Text</label>
      <textarea id="text" maxlength="2000">नमस्ते, मैं दिव्या हूँ। आज आप कैसे हैं?</textarea>
      <div style="margin-top: 12px;">
        <label for="model">Model</label>
        <select id="model"></select>
      </div>
      <div class="row">
        <div>
          <label for="language">Language</label>
          <select id="language"></select>
        </div>
        <div>
          <label for="voice">Voice</label>
          <select id="voice"></select>
        </div>
      </div>
      <button id="speak" type="submit">Speak</button>
      <div id="status" class="status"></div>
    </form>
    <section id="result" class="result">
      <label>Audio</label>
      <audio id="player" controls></audio>
    </section>
  </main>
  <script>
    const models = __MODELS__;
    const initial = __INITIAL__;
    const model = document.getElementById("model");
    const language = document.getElementById("language");
    const voice = document.getElementById("voice");
    const status = document.getElementById("status");
    const result = document.getElementById("result");
    const player = document.getElementById("player");
    const button = document.getElementById("speak");
    const title = document.getElementById("title");
    const lead = document.getElementById("lead");
    const text = document.getElementById("text");

    const samples = {
      hi: "नमस्ते, मैं दिव्या हूँ। आज आप कैसे हैं?",
      en: "Hello, this is Indic Parler-TTS on this Mac.",
      bn: "নমস্কার, আপনি কেমন আছেন?",
      mr: "नमस्कार, तुम्ही कसे आहात?",
      gu: "નમસ્તે, તમે કેમ છો?",
      pa: "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
      ta: "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
      te: "నమస్కారం, మీరు ఎలా ఉన్నారు?",
      kn: "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
      ml: "നമസ്കാരം, സുഖമാണോ?",
      or: "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?",
      as: "নমস্কাৰ, আপুনি কেনে আছে?",
      ne: "नमस्ते, तपाईं कस्तो हुनुहुन्छ?",
    };

    function currentModel() {
      return models.find((item) => item.id === model.value) || models[0];
    }

    function fillVoices() {
      const selected = currentModel().languages.find((item) => item.code === language.value)
        || currentModel().languages[0];
      voice.replaceChildren();
      for (const item of selected.voices) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.label;
        voice.appendChild(option);
      }
    }

    function fillLanguages() {
      const chosen = currentModel();
      title.textContent = chosen.name;
      lead.textContent = chosen.note;
      language.replaceChildren();
      for (const item of chosen.languages) {
        const option = document.createElement("option");
        option.value = item.code;
        option.textContent = item.name;
        language.appendChild(option);
      }
      fillVoices();
      if (samples[language.value]) text.value = samples[language.value];
    }

    for (const item of models) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.name;
      if (item.id === initial) option.selected = true;
      model.appendChild(option);
    }
    model.addEventListener("change", fillLanguages);
    language.addEventListener("change", () => {
      fillVoices();
      if (samples[language.value]) text.value = samples[language.value];
    });
    fillLanguages();

    document.getElementById("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const value = text.value.trim();
      if (!value) {
        status.textContent = "Enter some text first.";
        status.className = "status error";
        return;
      }
      button.disabled = true;
      status.className = "status";
      status.textContent = "Synthesizing (first run loads Indic Parler)…";
      try {
        const chosen = currentModel();
        const response = await fetch(chosen.route, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input: value,
            voice: voice.value,
            language: language.value,
          }),
        });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || response.statusText);
        }
        const blob = await response.blob();
        if (player.src.startsWith("blob:")) URL.revokeObjectURL(player.src);
        player.src = URL.createObjectURL(blob);
        result.classList.add("show");
        await player.play();
        status.textContent = "Ready.";
      } catch (error) {
        status.textContent = error.message || "Synthesis failed.";
        status.className = "status error";
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def render_page(initial: str = "indic-parler") -> bytes:
    html = PAGE.replace("__MODELS__", json.dumps(MODELS)).replace("__INITIAL__", json.dumps(initial))
    return html.encode()


class Handler(BaseHTTPRequestHandler):
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
        if path in ("/", "/index.html", "/indic-parler"):
            self._send(200, render_page("indic-parler"), "text/html; charset=utf-8")
            return
        if path == "/api/models":
            payload = {
                "device": engine.device,
                "models": [
                    {
                        "id": "indic-parler",
                        "repo": "ai4bharat/indic-parler-tts",
                        "route": "/api/indic-parler",
                        "sample_rate": 24000,
                        "languages": LANG_CATALOG,
                    }
                ],
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/indic-parler":
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
            voice = str(payload.get("voice") or "Divya")
            lang = str(payload.get("language") or "hi")
            audio = engine.synthesize(text, voice=voice, language=lang)
        except Exception as exc:  # noqa: BLE001
            self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "indic_parler.wav").write_bytes(audio)
        self._send(200, audio, "audio/wav")


def main() -> None:
    print("Indic Parler-TTS only (CosyVoice/Kokoro/Chatterbox not loaded).", flush=True)
    print(f"Open http://127.0.0.1:{UI_PORT}/", flush=True)
    print("Default: Hindi + Divya (female).", flush=True)
    ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
