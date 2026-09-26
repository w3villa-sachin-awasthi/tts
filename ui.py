"""Shared single-model TTS UI. Model choice is via run.py, not this page."""

from __future__ import annotations

import json
from typing import Any

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__PAGE_TITLE__</title>
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
    <h1 id="title">__PAGE_TITLE__</h1>
    <p class="lead" id="lead"></p>
    <form id="form">
      <label for="text">Text</label>
      <textarea id="text" maxlength="2000">__SAMPLE_TEXT__</textarea>
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
    const langSamples = __LANG_SAMPLES__;
    const loadingMessage = __LOADING_MSG__;
    const chosen = models.find((item) => item.id === initial) || models[0];
    const language = document.getElementById("language");
    const voice = document.getElementById("voice");
    const status = document.getElementById("status");
    const result = document.getElementById("result");
    const player = document.getElementById("player");
    const button = document.getElementById("speak");
    const text = document.getElementById("text");

    document.getElementById("title").textContent = chosen.name;
    document.getElementById("lead").textContent = chosen.note;

    function fillVoices() {
      const selected = chosen.languages.find((item) => item.code === language.value)
        || chosen.languages[0];
      voice.replaceChildren();
      for (const item of selected.voices) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.label;
        voice.appendChild(option);
      }
    }

    function fillLanguages() {
      language.replaceChildren();
      for (const item of chosen.languages) {
        const option = document.createElement("option");
        option.value = item.code;
        option.textContent = item.name;
        language.appendChild(option);
      }
      fillVoices();
      if (langSamples[language.value]) text.value = langSamples[language.value];
    }

    language.addEventListener("change", () => {
      fillVoices();
      if (langSamples[language.value]) text.value = langSamples[language.value];
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
      status.textContent = loadingMessage;
      try {
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


def render_page(
    *,
    page_title: str,
    models: list[dict[str, Any]],
    initial: str,
    sample_text: str,
    loading_message: str = "Synthesizing (first run loads weights)…",
    language_samples: dict[str, str] | None = None,
) -> bytes:
    samples = language_samples or {}
    html = _TEMPLATE
    html = html.replace("__PAGE_TITLE__", page_title)
    html = html.replace("__SAMPLE_TEXT__", sample_text)
    html = html.replace("__MODELS__", json.dumps(models))
    html = html.replace("__INITIAL__", json.dumps(initial))
    html = html.replace("__LANG_SAMPLES__", json.dumps(samples))
    html = html.replace("__LOADING_MSG__", json.dumps(loading_message))
    return html.encode()
