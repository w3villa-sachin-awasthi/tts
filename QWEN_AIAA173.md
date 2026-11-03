# AIAA-173 — Qwen voice clone on Jetson

Hardcoded in code (no `.env`):

| Constant | Value |
|----------|--------|
| `MODEL_NAME` | `qwen` |
| `VOICE_NAME` | `hr_clone` |
| `SAMPLE_RATE` | `24000` |
| `EMIT_MODE` | `stream` (latency); use `full` only if underrun intolerable |
| `STREAM_CHUNK_SIZE` | `8` (bench vs `4` / `12`) |
| `DTYPE_PREF` | `fp32` (try `bf16` separately; auto-fallback) |

Agent (`local_tts.py`): first TTS segment ~6–8 words; later ~12–16; FlushSentinel flushes if ≥6 words.

## Bench chunk size / dtype

```bash
cd /data/tts && source .venv/bin/activate
# Edit STREAM_CHUNK_SIZE=4|8|12 and/or DTYPE_PREF=fp32|bf16, then:
pm2 restart tts
python scripts/bench_qwen_sse.py
# Prefer highest RTF with acceptable TTFA. Orin NX often stays RTF<1.
```

## Run

```bash
pm2 restart tts
pm2 restart inbound-agent outbound-agent --update-env
curl -s http://127.0.0.1:8765/api/active
```
