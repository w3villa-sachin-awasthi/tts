"""Voice-only chat: Piper TTS + Ollama for replies. Speech input comes from the browser."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:270m")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. The user speaks aloud and hears your replies. "
    "Keep every answer short: one to three sentences, plain conversational English, no markdown."
)


def ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def list_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    names: list[str] = []
    for item in payload.get("models") or []:
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            names.append(name)
    return names


def resolve_ollama_model(preferred: str | None) -> str:
    preferred = (preferred or OLLAMA_MODEL).strip()
    models = list_ollama_models()
    if not models:
        return preferred
    if preferred in models:
        return preferred
    base = preferred.split(":", 1)[0]
    for name in models:
        if name == base or name.startswith(base + ":"):
            return name
    return models[0]


def chat_turn(history: list[dict], user_text: str, model: str) -> str:
    user_text = user_text.strip()
    if not user_text:
        raise ValueError("No speech was recognized.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    body = json.dumps(
        {"model": model, "messages": messages, "stream": False},
        ensure_ascii=False,
    ).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            err = json.loads(detail).get("error", detail)
        except json.JSONDecodeError:
            err = detail or exc.reason
        raise RuntimeError(f"Ollama error: {err}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        ) from exc
    message = payload.get("message") or {}
    reply = str(message.get("content", "")).strip()
    if not reply:
        raise RuntimeError("The language model returned an empty reply.")
    return reply


def turn_response(
    history: list[dict],
    user_text: str,
    voice: str,
    piper,
    ollama_model: str | None = None,
) -> dict:
    if not ollama_available():
        raise RuntimeError(
            "Ollama is not running. In a terminal run: ollama serve "
            f"(and pull a model, e.g. ollama pull {OLLAMA_MODEL})"
        )
    model = resolve_ollama_model(ollama_model)
    reply = chat_turn(history, user_text, model)
    audio = piper.synthesize(reply, voice)
    updated = list(history)
    updated.append({"role": "user", "content": user_text.strip()})
    updated.append({"role": "assistant", "content": reply})
    return {
        "user_text": user_text.strip(),
        "assistant_text": reply,
        "messages": updated,
        "voice": voice,
        "ollama_model": model,
        "audio_wav_base64": base64.b64encode(audio).decode("ascii"),
    }
