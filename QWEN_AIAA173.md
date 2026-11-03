# AIAA-173 — Qwen voice clone on Jetson

Hardcoded in code (no `.env`):

| Constant | Value |
|----------|--------|
| `MODEL_NAME` | `qwen` |
| `MODEL_ROUTE` | `/api/qwen` |
| `VOICE_NAME` | `hr_clone` |
| `LANGUAGE_NAME` | `English` |
| `SAMPLE_RATE` | `24000` |
| `STREAM_CHUNK_SIZE` | `12` (A/B vs `8`) |
| `EMIT_MODE` | `full` (smooth on RTF&lt;1; use `stream` for lower TTFA) |
| `DTYPE_PREF` | `fp32` (try `bf16` independently; auto-fallback) |

## Jetson files required

| Path | Purpose |
|------|---------|
| `/data/qwen3-tts/models/0.6B-Base` | Qwen 0.6B-Base weights (reuse; do not re-download) |
| `/data/qwen3-tts/ref/HR_VOICE.wav` | HR reference clip (used once to build embedding) |
| `/data/qwen3-tts/ref/hr_speaker.pt` | Cached x-vector (auto-created on first start) |

Engine: `faster-qwen3-tts`. Single-flight lock (no parallel Qwen). Agent buffers ~16 words before each POST.

## Run

```bash
pip install -U "faster-qwen3-tts" "numpy<2" "transformers==5.15.1"
# keep Jetson torch — do not overwrite with desktop wheels

pm2 restart tts
pm2 restart inbound-agent outbound-agent --update-env
curl -s http://127.0.0.1:8765/api/active
```

## Bench (chunk size / dtype / emit mode)

```bash
cd /data/tts && source .venv/bin/activate
# Edit qwen_engine.py: STREAM_CHUNK_SIZE=8|12, DTYPE_PREF=fp32|bf16, EMIT_MODE=full|stream
pm2 restart tts
python scripts/bench_qwen_sse.py
# Compare TTFA_ms and RTF. Prefer higher RTF without quality loss.
```

## Revert

```bash
git checkout <previous-branch>
pm2 restart tts inbound-agent outbound-agent
```
