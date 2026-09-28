#!/usr/bin/env bash
# Stop the TTS PM2 app, then you can delete the project folder safely.
set -euo pipefail

TTS_DIR="${1:-/data/tts}"

if command -v pm2 >/dev/null 2>&1; then
  pm2 delete tts 2>/dev/null || true
  pm2 save 2>/dev/null || true
  echo "PM2 app 'tts' removed."
else
  echo "pm2 not found; skip PM2 cleanup."
fi

echo ""
echo "Project data is only under: ${TTS_DIR}"
echo "  code, .venv, models, output — all inside that folder."
echo ""
echo "To wipe this app completely:"
echo "  rm -rf ${TTS_DIR}"
echo ""
echo "This does NOT uninstall Node, PM2, Ollama, or apt packages."
echo "Those are optional system tools; remove them only if you installed them for TTS."
