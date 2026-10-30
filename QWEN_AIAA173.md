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
| `/data/qwen3-tts/models/0.6B-Base` | Qwen 0.6B-Base weights |
| `/data/qwen3-tts/ref/HR_VOICE.wav` | HR reference clip |
| `REF_TEXT` in `qwen_engine.py` | Exact transcript of `HR_VOICE.wav` (hardcoded) |

## Run

TTS process needs Jetson torch 2.8 + torchaudio 2.8 + `qwen-tts` + `numpy<2`.

```bash
pm2 restart tts
pm2 restart inbound-agent outbound-agent
curl -s http://127.0.0.1:8765/api/active
```

## Revert

```bash
git checkout <previous-branch>   # both repos
pm2 restart tts inbound-agent outbound-agent
```
