# AIAA-173 — Qwen voice clone on Jetson

Hardcoded in code (no `.env`):

| Constant | Value |
|----------|--------|
| `MODEL_NAME` | `qwen` |
| `MODEL_ROUTE` | `/api/qwen` |
| `VOICE_NAME` | `hr_clone` |
| `LANGUAGE_NAME` | `English` |
| `SAMPLE_RATE` | `24000` |

## Jetson files required

| Path | Purpose |
|------|---------|
| `/data/qwen3-tts/models/0.6B-Base` | Qwen 0.6B-Base weights (reuse; do not re-download) |
| `/data/qwen3-tts/ref/HR_VOICE.wav` | HR reference clip (used once to build embedding) |
| `/data/qwen3-tts/ref/hr_speaker.pt` | Cached x-vector (auto-created on first start) |

Engine: `faster-qwen3-tts` (CUDA graphs + true streaming). Voice prompt is extracted once; per-request generation reuses `hr_speaker.pt`.

## Run

```bash
# TTS venv — do not overwrite Jetson torch
pip install -U "faster-qwen3-tts" "numpy<2"

pm2 restart tts
pm2 restart inbound-agent outbound-agent
curl -s http://127.0.0.1:8765/api/active
```

## Revert

```bash
git checkout <previous-branch>   # both repos
pm2 restart tts inbound-agent outbound-agent
```
