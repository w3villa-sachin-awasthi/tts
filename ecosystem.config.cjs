/**
 * Jetson / PM2 — everything for this app lives under /data/tts only.
 *
 * Start:  cd /data/tts && pm2 start ecosystem.config.cjs
 * Logs:    pm2 logs tts
 * Stop:    pm2 stop tts
 * Remove:  ./scripts/jetson-teardown.sh
 *
 * Voice chat needs Ollama separately (system install). It is NOT started here,
 * so deleting /data/tts does not leave an extra PM2 "ollama" process.
 */
module.exports = {
  apps: [
    {
      name: "tts",
      cwd: "/data/tts",
      script: ".venv/bin/python",
      args: "server.py",
      interpreter: "none",
      env: {
        TTS_HOST: "0.0.0.0",
        TTS_PORT: "8765",
        HF_HOME: "/data/tts/models/huggingface",
        HF_HUB_CACHE: "/data/tts/models/huggingface/hub",
        XDG_CACHE_HOME: "/data/tts/.cache",
        OLLAMA_URL: "http://127.0.0.1:11434",
        OLLAMA_MODEL: "gemma3:270m",
        // After: bash scripts/jetson-install-nemo-speech.sh
        NEMO_SPEECH_BIN: "/data/tts/nemo-speech/bin/nemo-speech",
        MAGPIE_DEVICE: "cuda",
        // Veena (maya-research/Veena) — set ACTIVE_TTS_MODEL=veena in .env to use
        VEENA_DEVICE: "cuda",
        VEENA_MODEL_ID: "maya-research/Veena",
        VEENA_LOAD_IN_4BIT: "0",
      },
      max_restarts: 10,
      restart_delay: 5000,
    },
  ],
};
