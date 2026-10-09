#!/usr/bin/env python3
"""Local TTS app. One ACTIVE_TTS_MODEL in RAM; all engines may remain on disk."""

from __future__ import annotations

import atexit
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Optional /data/tts/.env without requiring python-dotenv."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

MODEL_DIR = ROOT / "models" / "huggingface"
MAGPIE_MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
UI_HOST = os.environ.get("TTS_HOST", "127.0.0.1")
UI_PORT = int(os.environ.get("TTS_PORT", "8765"))
MAGPIE_PORT = 8091
MAGPIE_URL = f"http://127.0.0.1:{MAGPIE_PORT}"

os.environ.setdefault("HF_HOME", str(MODEL_DIR))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf

from active_tts import (
    ACTIVE_TTS_LANGUAGE,
    ACTIVE_TTS_MODEL,
    ACTIVE_TTS_VOICE,
    MODEL_SAMPLE_RATE,
    MODEL_TO_ROUTE,
    default_language_for,
    default_voice_for,
    detect_device,
    ensure_route_active,
    inactive_message,
    pin_language_error,
    pin_voice_error,
    write_env_hint,
)
from piper_engine import VOICES as PIPER_VOICES
from piper_engine import PiperEngine
from parler_engine import VOICES as PARLER_VOICES
from tts_stream import stream_kokoro, stream_parler, stream_piper

VOICE_CHAT_PAGE = (ROOT / "voice_chat.html").read_text(encoding="utf-8")

LANGUAGES = [
    (
        "a",
        "English (US)",
        [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "af_alloy",
            "af_aoede",
            "af_jessica",
            "af_kore",
            "af_nova",
            "af_river",
            "am_adam",
            "am_michael",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_onyx",
            "am_puck",
            "am_santa",
        ],
    ),
    (
        "b",
        "English (UK)",
        [
            "bf_emma",
            "bf_alice",
            "bf_isabella",
            "bf_lily",
            "bm_george",
            "bm_daniel",
            "bm_fable",
            "bm_lewis",
        ],
    ),
    ("e", "Spanish", ["ef_dora", "em_alex", "em_santa"]),
    ("f", "French", ["ff_siwis"]),
    ("h", "Hindi", ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]),
    ("i", "Italian", ["if_sara", "im_nicola"]),
    ("p", "Portuguese (Brazil)", ["pf_dora", "pm_alex", "pm_santa"]),
    (
        "j",
        "Japanese",
        ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    ),
    (
        "z",
        "Chinese",
        [
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
        ],
    ),
]


def voice_label(voice_id: str) -> str:
    kind = "female" if len(voice_id) > 1 and voice_id[1] == "f" else "male"
    name = voice_id.split("_", 1)[1].replace("_", " ").title()
    return f"{name} ({kind})"


def language_ready(code: str) -> bool:
    if code == "j":
        try:
            import pyopenjtalk  # noqa: F401
        except Exception:
            return False
    if code == "z":
        try:
            import jieba  # noqa: F401
        except Exception:
            return False
    return True


CATALOG = [
    {
        "code": code,
        "name": name,
        "voices": [{"id": voice, "label": voice_label(voice)} for voice in voices],
    }
    for code, name, voices in LANGUAGES
    if language_ready(code)
]
VOICE_LANG = {
    voice["id"]: item["code"] for item in CATALOG for voice in item["voices"]
}

DEVICE = detect_device() if ACTIVE_TTS_MODEL in ("kokoro", "speecht5", "parler") else "cpu"
pipelines: dict[str, object] = {}
_kokoro_model = None
pipeline_lock = threading.Lock()
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))
KOKORO_SPLIT = r"[.!?\u0964]\s+"
_speecht5 = None
_parler = None
_piper: PiperEngine | None = None
magpie_proc: subprocess.Popen | None = None
magpie_lock = threading.Lock()

# Catalog metadata only (no model weights / no torch import).
SPEECHT5_SPEAKERS = [
    ("slt", "SLT (US female)"),
    ("clb", "CLB (US female)"),
    ("bdl", "BDL (US male)"),
    ("rms", "RMS (US male)"),
    ("ksp", "KSP (Indian English)"),
    ("awb", "AWB (Scottish)"),
]
SPEECHT5_LANG = [
    {
        "code": "en",
        "name": "English",
        "voices": [{"id": key, "label": label} for key, label in SPEECHT5_SPEAKERS],
    }
]
MAGPIE_VOICES = [
    {"id": name, "label": name} for name in ("Aria", "Jason", "John", "Leo", "Sofia")
]
MAGPIE_LANG = [
    {"code": code, "name": name, "voices": MAGPIE_VOICES}
    for code, name in (
        ("en-US", "English"),
        ("es-ES", "Spanish"),
        ("de-DE", "German"),
        ("fr-FR", "French"),
        ("it-IT", "Italian"),
        ("vi-VN", "Vietnamese"),
        ("hi-IN", "Hindi"),
    )
]
PIPER_LANG = []
for code, name, voice_id, label in PIPER_VOICES:
    match = next((item for item in PIPER_LANG if item["code"] == code), None)
    if match is None:
        match = {"code": code, "name": name, "voices": []}
        PIPER_LANG.append(match)
    match["voices"].append({"id": voice_id, "label": label})

PARLER_LANG = [
    {
        "code": "en",
        "name": "Indian English",
        "voices": [{"id": name, "label": f"{name} (female)"} for name in PARLER_VOICES],
    }
]

MODELS = [
    {
        "id": "kokoro",
        "name": "Kokoro-82M",
        "route": "/api/speech",
        "note": "Multilingual. The voice has to match the language.",
        "streaming": True,
        "languages": CATALOG,
    },
    {
        "id": "speecht5",
        "name": "SpeechT5",
        "route": "/api/speecht5",
        "note": "English only. This checkpoint was trained on LibriTTS.",
        "languages": SPEECHT5_LANG,
    },
    {
        "id": "magpie",
        "name": "Magpie TTS 357M",
        "route": "/api/magpie",
        "note": "357 million parameters, not billion. v2602 on the M2 GPU.",
        "languages": MAGPIE_LANG,
    },
    {
        "id": "piper",
        "name": "Piper ~20M",
        "route": "/api/piper",
        "note": "Medium Piper voices, about 15–20 million parameters each.",
        "streaming": True,
        "languages": PIPER_LANG,
    },
    {
        "id": "parler",
        "name": "Indic Parler-TTS",
        "route": "/api/parler",
        "note": "Indian English female voices (Mary default).",
        "streaming": True,
        "languages": PARLER_LANG,
    },
]


def get_piper() -> PiperEngine:
    global _piper
    if ACTIVE_TTS_MODEL != "piper":
        raise RuntimeError(inactive_message("piper"))
    if _piper is None:
        _piper = PiperEngine(ROOT / "models" / "piper", single_active=True)
    return _piper


def get_speecht5():
    global _speecht5, DEVICE
    if ACTIVE_TTS_MODEL != "speecht5":
        raise RuntimeError(inactive_message("speecht5"))
    if _speecht5 is None:
        from speecht5_engine import SpeechT5Engine

        DEVICE = detect_device()
        _speecht5 = SpeechT5Engine(DEVICE)
    return _speecht5


def get_parler():
    global _parler, DEVICE
    if ACTIVE_TTS_MODEL != "parler":
        raise RuntimeError(inactive_message("parler"))
    if _parler is None:
        from parler_engine import ParlerEngine

        DEVICE = detect_device()
        _parler = ParlerEngine(DEVICE)
    return _parler


def pipeline_for(lang: str):
    """Kokoro: one model in RAM, shared by every language pipeline (e.g. English + Hindi)."""
    global DEVICE, _kokoro_model
    if ACTIVE_TTS_MODEL != "kokoro":
        raise RuntimeError(inactive_message("kokoro"))
    pin_err = pin_language_error(lang)
    if pin_err:
        raise RuntimeError(pin_err)
    if lang in pipelines:
        return pipelines[lang]
    from kokoro import KModel, KPipeline

    if _kokoro_model is None:
        DEVICE = detect_device()
        _kokoro_model = KModel().to(DEVICE).eval()
    pipelines[lang] = KPipeline(lang_code=lang, model=_kokoro_model, device=DEVICE)
    return pipelines[lang]


def preload_active_engine() -> None:
    """Load only ACTIVE_TTS_MODEL into RAM. Other engines stay on disk unused."""
    voice = default_voice_for(ACTIVE_TTS_MODEL)
    lang = default_language_for(ACTIVE_TTS_MODEL)
    print(
        f"Active TTS: {ACTIVE_TTS_MODEL} (disk cache under {ROOT / 'models'}; "
        "only this engine loads into RAM)",
        flush=True,
    )
    print(write_env_hint(ROOT), flush=True)
    if ACTIVE_TTS_MODEL == "kokoro":
        print(f"Preloading Kokoro lang={lang} voice={voice} on {detect_device()}…", flush=True)
        pipeline_for(lang)
    elif ACTIVE_TTS_MODEL == "piper":
        print(f"Preloading Piper voice={voice}…", flush=True)
        get_piper().voice(voice)
    elif ACTIVE_TTS_MODEL == "speecht5":
        print(f"Preloading SpeechT5 on {detect_device()}…", flush=True)
        get_speecht5().load()
    elif ACTIVE_TTS_MODEL == "magpie":
        print("Starting Magpie subprocess…", flush=True)
        start_magpie()
    elif ACTIVE_TTS_MODEL == "parler":
        print(f"Preloading Parler voice={voice} on {detect_device()}…", flush=True)
        get_parler().load()


def _native_binary(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(4)
    except OSError:
        return False
    if sys.platform == "linux":
        return header.startswith(b"\x7fELF")
    if sys.platform == "darwin":
        return header in (
            b"\xfe\xed\xfa\xce",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xcf\xfa\xed\xfe",
        )
    return True


def resolve_nemo_bin() -> Path | None:
    override = os.environ.get("NEMO_SPEECH_BIN", "").strip()
    if override:
        path = Path(override)
        return path if _native_binary(path) else None
    for path in (
        ROOT / "nemo-speech" / "bin" / "nemo-speech",
        Path.home() / ".local" / "bin" / "nemo-speech",
        ROOT / "vendor" / "nemo-speech" / "bin" / "nemo-speech",
    ):
        if _native_binary(path):
            return path
    return None


def magpie_device() -> str:
    override = os.environ.get("MAGPIE_DEVICE", "").strip().lower()
    if override:
        return override
    if sys.platform == "darwin":
        return "metal"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def magpie_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{MAGPIE_URL}/health", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def start_magpie() -> None:
    global magpie_proc
    if magpie_ready():
        return
    nemo = resolve_nemo_bin()
    if nemo is None:
        raise RuntimeError(
            "NeMo-Speech.cpp is not installed for this CPU/OS. "
            "The vendor/ copy in git is macOS-only. On Jetson run: "
            "bash scripts/jetson-install-nemo-speech.sh "
            "then pm2 restart tts --update-env"
        )
    with magpie_lock:
        if magpie_ready():
            return
        if magpie_proc and magpie_proc.poll() is None:
            pass
        else:
            env = os.environ.copy()
            env["NEMO_SPEECH_MODEL_DIR"] = str(MAGPIE_MODEL_DIR)
            magpie_proc = subprocess.Popen(
                [
                    str(nemo),
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(MAGPIE_PORT),
                    "--tts-model",
                    "magpie",
                    "--device",
                    magpie_device(),
                    "--no-ui",
                    "--tts.voice-name",
                    "Sofia",
                ],
                cwd=ROOT,
                env=env,
            )
        deadline = time.time() + 180
        while time.time() < deadline:
            if magpie_proc and magpie_proc.poll() is not None:
                raise RuntimeError(f"Magpie exited with code {magpie_proc.returncode}")
            if magpie_ready():
                return
            time.sleep(0.4)
        raise RuntimeError("Magpie did not become ready.")


def stop_magpie() -> None:
    if magpie_proc and magpie_proc.poll() is None:
        magpie_proc.terminate()


atexit.register(stop_magpie)


def synthesize_magpie(text: str, voice: str, lang: str) -> bytes:
    if voice not in {item["id"] for item in MAGPIE_VOICES}:
        raise ValueError("Unknown Magpie voice.")
    if lang not in {item["code"] for item in MAGPIE_LANG}:
        raise ValueError("That language is not in this Magpie build.")
    start_magpie()
    body = json.dumps(
        {
            "model": "magpie",
            "input": text,
            "voice": voice,
            "language": lang,
            "response_format": "wav",
        }
    ).encode()
    request = urllib.request.Request(
        f"{MAGPIE_URL}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(detail or exc.reason) from exc


def synthesize(text: str, voice: str, lang: str) -> bytes:
    with pipeline_lock:
        chunks = []
        for _graphemes, _phonemes, audio in pipeline_for(lang)(
            text, voice=voice, speed=KOKORO_SPEED, split_pattern=KOKORO_SPLIT
        ):
            if audio is not None and len(audio):
                chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        raise RuntimeError("Kokoro returned no audio.")
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(chunks), 24000, format="WAV")
    return buffer.getvalue()


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Local TTS</title>
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
    <h1 id="title">Kokoro-82M</h1>
    <p class="lead" id="lead">Local speech on this Mac.</p>
    <form id="form">
      <label for="text">Text</label>
      <textarea id="text" maxlength="2000">Hello from Kokoro on this Mac.</textarea>
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
      <label style="display:block;margin-top:12px;font-size:0.9rem;">
        <input type="checkbox" id="stream" /> Stream audio (Kokoro, Piper, Parler)
      </label>
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
    const streamBox = document.getElementById("stream");
    const title = document.getElementById("title");
    const lead = document.getElementById("lead");
    let streamCtx = null;
    let streamNextTime = 0;

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

    const samples = {
      kokoro: "Hello from Kokoro on this Mac.",
      parler: "Hello, how can I help you today?",
      speecht5: "Hello from SpeechT5 on this Mac.",
      magpie: "Hello from Magpie on this Mac.",
      piper: "Hello from Piper on this Mac.",
    };
    const paths = { kokoro: "/", speecht5: "/speecht5", magpie: "/magpie", piper: "/piper", parler: "/parler" };

    function syncStreamToggle() {
      const canStream = !!currentModel().streaming;
      streamBox.disabled = !canStream;
      streamBox.parentElement.style.opacity = canStream ? "1" : "0.5";
      if (!canStream) streamBox.checked = false;
    }

    function parseSseBlock(block) {
      let event = "message";
      let data = "";
      for (const line of block.split("\\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data = line.slice(5).trim();
      }
      return { event, data: data ? JSON.parse(data) : {} };
    }

    async function playStreamingResponse(response) {
      if (streamCtx) {
        await streamCtx.close();
        streamCtx = null;
      }
      streamCtx = new AudioContext();
      streamNextTime = streamCtx.currentTime;
      let sampleRate = 24000;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        pending += decoder.decode(value, { stream: true });
        let split;
        while ((split = pending.indexOf("\\n\\n")) >= 0) {
          const raw = pending.slice(0, split);
          pending = pending.slice(split + 2);
          const { event, data } = parseSseBlock(raw);
          if (event === "meta" && data.sample_rate) sampleRate = data.sample_rate;
          if (event === "error") throw new Error(data.message || "Stream error");
          if (event === "chunk" && data.b64) {
            const rate = data.sample_rate || sampleRate;
            const binary = atob(data.b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            const floats = new Float32Array(bytes.buffer);
            const buffer = streamCtx.createBuffer(1, floats.length, rate);
            buffer.copyToChannel(floats, 0);
            const source = streamCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(streamCtx.destination);
            const start = Math.max(streamNextTime, streamCtx.currentTime);
            source.start(start);
            streamNextTime = start + buffer.duration;
          }
        }
      }
      result.classList.add("show");
      player.removeAttribute("src");
    }

    function fillLanguages() {
      const chosen = currentModel();
      title.textContent = chosen.name;
      lead.textContent = chosen.note;
      syncStreamToggle();
      const text = document.getElementById("text");
      if (Object.values(samples).includes(text.value)) text.value = samples[chosen.id];
      language.replaceChildren();
      for (const item of chosen.languages) {
        const option = document.createElement("option");
        option.value = item.code;
        option.textContent = item.name;
        language.appendChild(option);
      }
      fillVoices();
    }

    for (const item of models) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.name;
      if (item.id === initial) option.selected = true;
      model.appendChild(option);
    }
    model.addEventListener("change", () => {
      history.replaceState(null, "", paths[model.value] || "/");
      fillLanguages();
    });
    language.addEventListener("change", fillVoices);
    fillLanguages();

    document.getElementById("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = document.getElementById("text").value.trim();
      if (!text) {
        status.textContent = "Enter some text first.";
        status.className = "status error";
        return;
      }
      button.disabled = true;
      status.className = "status";
      const useStream = streamBox.checked && currentModel().streaming;
      status.textContent = useStream ? "Streaming…" : "Synthesizing…";
      try {
        const chosen = currentModel();
        const response = await fetch(chosen.route, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input: text,
            voice: voice.value,
            language: language.value,
            stream: useStream,
          }),
        });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || response.statusText);
        }
        if (useStream) {
          await playStreamingResponse(response);
          status.textContent = "Stream finished.";
        } else {
          const blob = await response.blob();
          if (player.src.startsWith("blob:")) URL.revokeObjectURL(player.src);
          player.src = URL.createObjectURL(blob);
          result.classList.add("show");
          await player.play();
          status.textContent = "Ready.";
        }
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


def render_page(initial: str) -> bytes:
    # UI only exposes the RAM-active engine; other models stay on disk until restart.
    active_models = [item for item in MODELS if item["id"] == ACTIVE_TTS_MODEL]
    if not active_models:
        active_models = MODELS[:1]
    if initial != ACTIVE_TTS_MODEL:
        initial = ACTIVE_TTS_MODEL
    html = PAGE.replace("__MODELS__", json.dumps(active_models)).replace(
        "__INITIAL__", json.dumps(initial)
    )
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

    def _send_sse(self, chunks) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for block in chunks:
            self.wfile.write(block)
            self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/kokoro"):
            self._send(200, render_page("kokoro"), "text/html; charset=utf-8")
            return
        if path == "/speecht5":
            self._send(200, render_page("speecht5"), "text/html; charset=utf-8")
            return
        if path == "/magpie":
            self._send(200, render_page("magpie"), "text/html; charset=utf-8")
            return
        if path == "/piper":
            self._send(200, render_page("piper"), "text/html; charset=utf-8")
            return
        if path == "/parler":
            self._send(200, render_page("parler"), "text/html; charset=utf-8")
            return
        if path == "/voice-chat":
            self._send(200, VOICE_CHAT_PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/models":
            catalog_by_id = {
                "kokoro": {
                    "id": "kokoro",
                    "repo": "hexgrad/Kokoro-82M",
                    "route": "/api/speech",
                    "sample_rate": 24000,
                    "streaming": True,
                    "languages": CATALOG,
                },
                "speecht5": {
                    "id": "speecht5",
                    "repo": "microsoft/speecht5_tts",
                    "route": "/api/speecht5",
                    "sample_rate": 16000,
                    "languages": SPEECHT5_LANG,
                },
                "magpie": {
                    "id": "magpie",
                    "repo": "nvidia/magpie_tts_multilingual_357m",
                    "route": "/api/magpie",
                    "sample_rate": 22050,
                    "languages": MAGPIE_LANG,
                },
                "piper": {
                    "id": "piper",
                    "repo": "rhasspy/piper-voices",
                    "route": "/api/piper",
                    "sample_rate": 22050,
                    "streaming": True,
                    "languages": PIPER_LANG,
                },
                "parler": {
                    "id": "parler",
                    "repo": "ai4bharat/indic-parler-tts",
                    "route": "/api/parler",
                    "sample_rate": MODEL_SAMPLE_RATE["parler"],
                    "streaming": True,
                    "languages": PARLER_LANG,
                },
            }
            models = []
            for mid, item in catalog_by_id.items():
                entry = dict(item)
                entry["active"] = mid == ACTIVE_TTS_MODEL
                entry["loaded_in_ram"] = mid == ACTIVE_TTS_MODEL
                models.append(entry)
            payload = {
                "device": DEVICE,
                "active_model": ACTIVE_TTS_MODEL,
                "active_voice": ACTIVE_TTS_VOICE or None,
                "active_language": ACTIVE_TTS_LANGUAGE or None,
                "active_route": MODEL_TO_ROUTE[ACTIVE_TTS_MODEL],
                "active_sample_rate": MODEL_SAMPLE_RATE[ACTIVE_TTS_MODEL],
                "policy": (
                    "All model files may remain on disk under ./models. "
                    "Only ACTIVE_TTS_MODEL is loaded into RAM. "
                    "Change ACTIVE_TTS_* in /data/tts/.env then: pm2 restart tts --update-env"
                ),
                "models": models,
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        if path == "/api/active":
            payload = {
                "active_model": ACTIVE_TTS_MODEL,
                "active_voice": ACTIVE_TTS_VOICE or default_voice_for(ACTIVE_TTS_MODEL),
                "active_language": ACTIVE_TTS_LANGUAGE or default_language_for(ACTIVE_TTS_MODEL),
                "route": MODEL_TO_ROUTE[ACTIVE_TTS_MODEL],
                "sample_rate": MODEL_SAMPLE_RATE[ACTIVE_TTS_MODEL],
                "device": DEVICE,
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0"))
        if path == "/api/voice-chat/turn":
            if ACTIVE_TTS_MODEL != "piper":
                self._send(503, inactive_message("piper").encode(), "text/plain; charset=utf-8")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode())
            except json.JSONDecodeError:
                self._send(400, b"Invalid JSON", "text/plain; charset=utf-8")
                return
            user_text = str(payload.get("user_text", "")).strip()
            voice = str(payload.get("voice") or default_voice_for("piper"))
            pin_err = pin_voice_error(voice)
            if pin_err:
                self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                return
            history = payload.get("messages") or []
            if not isinstance(history, list):
                self._send(400, b"messages must be a list", "text/plain; charset=utf-8")
                return
            ollama_model = payload.get("ollama_model")
            if ollama_model is not None:
                ollama_model = str(ollama_model).strip() or None
            try:
                from voice_chat import turn_response

                result = turn_response(
                    history, user_text, voice, get_piper(), ollama_model=ollama_model
                )
            except Exception as exc:  # noqa: BLE001
                self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
                return
            self._send(200, json.dumps(result).encode(), "application/json")
            return
        if path not in ("/api/speech", "/api/speecht5", "/api/magpie", "/api/piper", "/api/parler"):
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        inactive = ensure_route_active(path)
        if inactive:
            self._send(503, inactive.encode(), "text/plain; charset=utf-8")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode())
        except json.JSONDecodeError:
            self._send(400, b"Invalid JSON", "text/plain; charset=utf-8")
            return
        text = str(payload.get("input", "")).strip()
        if not text:
            self._send(400, b"Text is required", "text/plain; charset=utf-8")
            return
        want_stream = bool(payload.get("stream"))
        if want_stream and path not in ("/api/speech", "/api/piper", "/api/parler"):
            self._send(
                400,
                b"Streaming is only supported for Kokoro, Piper, and Parler.",
                "text/plain; charset=utf-8",
            )
            return
        if want_stream and path == "/api/piper":
            voice = str(payload.get("voice") or default_voice_for("piper"))
            pin_err = pin_voice_error(voice)
            if pin_err:
                self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                return
            try:
                self._send_sse(stream_piper(text, voice, get_piper()))
            except Exception as exc:  # noqa: BLE001
                self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
            return
        if want_stream and path == "/api/parler":
            voice = str(payload.get("voice") or default_voice_for("parler"))
            lang = str(payload.get("language") or default_language_for("parler"))
            pin_err = pin_voice_error(voice) or pin_language_error(lang)
            if pin_err:
                self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                return
            try:
                self._send_sse(stream_parler(text, voice, get_parler()))
            except Exception as exc:  # noqa: BLE001
                self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
            return
        if want_stream:
            voice = str(payload.get("voice") or default_voice_for("kokoro"))
            lang = str(payload.get("language") or VOICE_LANG.get(voice, "a"))
            pin_err = pin_voice_error(voice) or pin_language_error(lang)
            if pin_err:
                self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                return
            try:
                self._send_sse(
                    stream_kokoro(
                        text,
                        voice,
                        lang,
                        pipeline_for,
                        pipeline_lock,
                        VOICE_LANG,
                        KOKORO_SPEED,
                        KOKORO_SPLIT,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
            return
        try:
            if path == "/api/speecht5":
                voice = str(payload.get("voice") or default_voice_for("speecht5"))
                pin_err = pin_voice_error(voice)
                if pin_err:
                    self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                    return
                audio = get_speecht5().synthesize(text, voice)
                filename = "speecht5.wav"
            elif path == "/api/magpie":
                voice = str(payload.get("voice") or default_voice_for("magpie"))
                lang = str(payload.get("language") or default_language_for("magpie"))
                pin_err = pin_voice_error(voice) or pin_language_error(lang)
                if pin_err:
                    self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                    return
                audio = synthesize_magpie(text, voice, lang)
                filename = "magpie.wav"
            elif path == "/api/piper":
                voice = str(payload.get("voice") or default_voice_for("piper"))
                pin_err = pin_voice_error(voice)
                if pin_err:
                    self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                    return
                audio = get_piper().synthesize(text, voice)
                filename = "piper.wav"
            elif path == "/api/parler":
                voice = str(payload.get("voice") or default_voice_for("parler"))
                lang = str(payload.get("language") or default_language_for("parler"))
                pin_err = pin_voice_error(voice) or pin_language_error(lang)
                if pin_err:
                    self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                    return
                audio = get_parler().synthesize(text, voice)
                filename = "parler.wav"
            else:
                voice = str(payload.get("voice") or default_voice_for("kokoro"))
                lang = str(payload.get("language") or VOICE_LANG.get(voice, "a"))
                pin_err = pin_voice_error(voice) or pin_language_error(lang)
                if pin_err:
                    self._send(400, pin_err.encode(), "text/plain; charset=utf-8")
                    return
                if voice not in VOICE_LANG:
                    self._send(400, b"Unknown Kokoro voice.", "text/plain; charset=utf-8")
                    return
                audio = synthesize(text, voice, lang)
                filename = "kokoro.wav"
        except Exception as exc:  # noqa: BLE001
            self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / filename).write_bytes(audio)
        self._send(200, audio, "audio/wav")


def main() -> None:
    preload_active_engine()
    base = f"http://{UI_HOST}:{UI_PORT}" if UI_HOST not in ("0.0.0.0", "::") else f"http://127.0.0.1:{UI_PORT}"
    print(f"Listening on {UI_HOST}:{UI_PORT}", flush=True)
    print(f"Active route: {base}{MODEL_TO_ROUTE[ACTIVE_TTS_MODEL]}", flush=True)
    print(f"Status: {base}/api/active  |  catalog: {base}/api/models", flush=True)
    if ACTIVE_TTS_MODEL == "piper":
        print(f"UI: {base}/piper", flush=True)
    elif ACTIVE_TTS_MODEL == "kokoro":
        print(f"UI: {base}/", flush=True)
    elif ACTIVE_TTS_MODEL == "speecht5":
        print(f"UI: {base}/speecht5", flush=True)
    elif ACTIVE_TTS_MODEL == "magpie":
        print(f"UI: {base}/magpie", flush=True)
    elif ACTIVE_TTS_MODEL == "parler":
        print(f"UI: {base}/parler", flush=True)
    ThreadingHTTPServer((UI_HOST, UI_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
