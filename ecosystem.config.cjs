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
        OLLAMA_URL: "http://127.0.0.1:11434",
        OLLAMA_MODEL: "gemma3:270m",
      },
    },
  ],
};
