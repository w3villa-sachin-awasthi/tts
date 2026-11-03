# Chatterbox Multilingual V3 (Jetson)

Hardcoded switch in `active_tts.py`:


| Constant        | Value                                                                  |
| --------------- | ---------------------------------------------------------------------- |
| `MODEL_NAME`    | `chatterbox`                                                           |
| `VOICE_NAME`    | `ref_clone`                                                            |
| `LANGUAGE_NAME` | `` (empty — UI/API may send `en` or `hi`)                              |
| Paths           | `/data/chatterbox/ref/voice.wav`, HF cache `/data/chatterbox/hf-cache` |


**Streaming:** official API is full-waveform `generate()` only. Do not enable SSE for this model.

## Install (new venv — do not touch Qwen)

```bash
sudo mkdir -p /data/chatterbox/{ref,hf-cache,logs}
cd /data/tts   # or clone this repo here
python3 -m venv /data/chatterbox/venv
source /data/chatterbox/venv/bin/activate
pip install -U pip wheel
# Jetson torch FIRST — never desktop PyPI torch
pip install torch torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install soundfile colorlog fastapi uvicorn python-multipart
pip install chatterbox-tts
# If pip overwrote torch, reinstall Jetson wheels again:
# pip install torch torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Put a 5–15s clean WAV at `/data/chatterbox/ref/voice.wav`, then in `active_tts.py` activate the chatterbox block, restart:

```bash
export HF_HOME=/data/chatterbox/hf-cache
export HUGGINGFACE_HUB_CACHE=/data/chatterbox/hf-cache
cd /data/tts && source /data/chatterbox/venv/bin/activate
# Use a different port if Qwen already owns 8765:
TTS_HOST=0.0.0.0 TTS_PORT=8766 python server.py
```



## API

- `GET /health` — model / device
- `GET /api/active` — active engine
- `POST /api/chatterbox/voice` — raw WAV body → saves `ref/voice.wav`
- `POST /api/chatterbox` — JSON `{input, language: en|hi, voice: ref_clone}` → WAV

```bash
curl -s http://127.0.0.1:8766/health
curl -s -X POST http://127.0.0.1:8766/api/chatterbox/voice \
  --data-binary @/data/chatterbox/ref/voice.wav -H 'Content-Type: audio/wav'
curl -s -X POST http://127.0.0.1:8766/api/chatterbox \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Jetson.","language":"en","voice":"ref_clone"}' \
  --output /tmp/cb_en.wav
python scripts/bench_chatterbox.py http://127.0.0.1:8766
```

