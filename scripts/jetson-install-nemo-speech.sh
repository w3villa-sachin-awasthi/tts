#!/usr/bin/env bash
# Linux/Jetson NeMo-Speech for Magpie — installs under /data/tts/nemo-speech only.
set -euo pipefail

ROOT="${1:-/data/tts}"
PREFIX="${ROOT}/nemo-speech"
MODEL_DIR="${ROOT}/models"

echo "Installing NeMo-Speech.cpp to ${PREFIX} (Orin / CUDA 12)…"
export NEMO_SPEECH_CUDA_SERIES="${NEMO_SPEECH_CUDA_SERIES:-12}"

curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/main/scripts/install.sh |
  sh -s -- --prefix "${PREFIX}" --backend cuda

export PATH="${PREFIX}/bin:${PATH}"
export NEMO_SPEECH_MODEL_DIR="${MODEL_DIR}"

echo "Pulling Magpie weights into ${MODEL_DIR} (may take a while)…"
"${PREFIX}/bin/nemo-speech" pull magpie

echo ""
echo "Done. Add to PM2 ecosystem.config.cjs env (or export before pm2 start):"
echo "  NEMO_SPEECH_BIN=${PREFIX}/bin/nemo-speech"
echo "  MAGPIE_DEVICE=cuda"
echo "Then: pm2 restart tts --update-env"
