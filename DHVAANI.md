# DhVaani-0.5 (Jetson) — separate PM2 from Goonj

Layout (same idea as Chatterbox):

| Path | Role |
|------|------|
| `/data/dhvaani/venv` | own Python deps |
| `/data/dhvaani/ref/voice.wav` | 3–10 s clean ref (clone voice) |
| `/data/dhvaani/ref/prompt.txt` | **exact transcript** of that WAV |
| `/data/dhvaani/hf-cache` | HF weights |
| Code | `/data/tts` (this repo) |

Model: [ARTPARK-IISc/DhVaani-0.5](https://huggingface.co/ARTPARK-IISc/DhVaani-0.5)  
Optional later speedup: [TRT aarch64](https://huggingface.co/smajji/DhVaani-0.5-TensorRT-aarch64)

Voice clone: pin `voice.wav` + `prompt.txt` once at warm; every synth reuses that prompt (no re-upload).

## Install on Jetson

```bash
sudo mkdir -p /data/dhvaani/{ref,hf-cache,logs}
python3 -m venv /data/dhvaani/venv
source /data/dhvaani/venv/bin/activate
pip install -U pip wheel
# Jetson torch FIRST
pip install torch torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install soundfile colorlog "numpy<2" transformers accelerate safetensors huggingface_hub
# if pip overwrote torch, reinstall Jetson wheels again

# download weights into hf-cache (needs network; gated? set HF_TOKEN if asked)
export HF_HOME=/data/dhvaani/hf-cache
export HUGGINGFACE_HUB_CACHE=/data/dhvaani/hf-cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ARTPARK-IISc/DhVaani-0.5', trust_remote_code=True, cache_dir='/data/dhvaani/hf-cache')"
```

Put ref files:

```bash
# scp your 3–10s WAV → /data/dhvaani/ref/voice.wav
printf '%s\n' 'exact words spoken in voice.wav' > /data/dhvaani/ref/prompt.txt
```

## Run (do not touch Goonj `tts` PM2)

```bash
cd /data/tts
source /data/dhvaani/venv/bin/activate
export HF_HOME=/data/dhvaani/hf-cache
export HUGGINGFACE_HUB_CACHE=/data/dhvaani/hf-cache
export TTS_MODEL_NAME=dhvaani
export TTS_VOICE_NAME=ref_clone
export TTS_LANGUAGE_NAME=hi
TTS_HOST=0.0.0.0 TTS_PORT=8767 python server.py
```

PM2 example:

```bash
pm2 start /data/dhvaani/venv/bin/python \
  --name tts-dhvaani \
  --cwd /data/tts \
  --interpreter none \
  -- \
  server.py
# then: pm2 set env for that app, or use an ecosystem file with:
#   TTS_HOST=0.0.0.0 TTS_PORT=8767 TTS_MODEL_NAME=dhvaani
#   HF_HOME=/data/dhvaani/hf-cache HUGGINGFACE_HUB_CACHE=/data/dhvaani/hf-cache
```

## API

```bash
curl -s http://127.0.0.1:8767/health
curl -s -X POST http://127.0.0.1:8767/api/dhvaani/voice \
  -H 'Content-Type: application/json' \
  -d "{\"prompt_text\":\"exact transcript of ref\",\"wav_b64\":\"$(base64 -w0 /data/dhvaani/ref/voice.wav)\"}"
curl -s -X POST http://127.0.0.1:8767/api/dhvaani \
  -H 'Content-Type: application/json' \
  -d '{"input":"नमस्ते, आपकी अपॉइंटमेंट कन्फर्म हो गई है।","language":"hi","voice":"ref_clone"}' \
  --output /tmp/dhvaani.wav
```

Calling agent: in `local_tts.py` point `TTS_BASE_URL` at `8767` and uncomment the dhvaani block.
