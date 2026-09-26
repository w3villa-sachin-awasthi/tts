"""hexgrad Kokoro-82M. Multilingual, loaded on first use only."""

from __future__ import annotations

import io
import threading

import numpy as np
import soundfile as sf
import torch

LANGUAGES = [
    (
        "a",
        "English (US)",
        [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "af_alloy",
            "af_aoede",
            "af_jessica",
            "af_kore",
            "af_nova",
            "af_river",
            "am_adam",
            "am_michael",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_onyx",
            "am_puck",
            "am_santa",
        ],
    ),
    (
        "b",
        "English (UK)",
        [
            "bf_emma",
            "bf_alice",
            "bf_isabella",
            "bf_lily",
            "bm_george",
            "bm_daniel",
            "bm_fable",
            "bm_lewis",
        ],
    ),
    ("e", "Spanish", ["ef_dora", "em_alex", "em_santa"]),
    ("f", "French", ["ff_siwis"]),
    ("h", "Hindi", ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]),
    ("i", "Italian", ["if_sara", "im_nicola"]),
    ("p", "Portuguese (Brazil)", ["pf_dora", "pm_alex", "pm_santa"]),
    (
        "j",
        "Japanese",
        ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    ),
    (
        "z",
        "Chinese",
        [
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
        ],
    ),
]

SAMPLE_RATE = 24000


def voice_label(voice_id: str) -> str:
    kind = "female" if len(voice_id) > 1 and voice_id[1] == "f" else "male"
    name = voice_id.split("_", 1)[1].replace("_", " ").title()
    return f"{name} ({kind})"


def language_ready(code: str) -> bool:
    if code == "j":
        try:
            import pyopenjtalk  # noqa: F401
        except Exception:
            return False
    if code == "z":
        try:
            import jieba  # noqa: F401
        except Exception:
            return False
    return True


def build_catalog() -> list[dict]:
    return [
        {
            "code": code,
            "name": name,
            "voices": [{"id": voice, "label": voice_label(voice)} for voice in voices],
        }
        for code, name, voices in LANGUAGES
        if language_ready(code)
    ]


class KokoroEngine:
    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.lock = threading.Lock()
        self.pipelines: dict[str, object] = {}

    def pipeline_for(self, lang: str):
        """Caller must hold self.lock."""
        from kokoro import KPipeline

        if lang not in self.pipelines:
            self.pipelines[lang] = KPipeline(lang_code=lang, device=self.device)
        return self.pipelines[lang]

    def synthesize(self, text: str, voice: str, lang: str) -> bytes:
        with self.lock:
            chunks = []
            for _graphemes, _phonemes, audio in self.pipeline_for(lang)(
                text, voice=voice, speed=1
            ):
                if audio is not None and len(audio):
                    chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")
        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(chunks), SAMPLE_RATE, format="WAV")
        return buffer.getvalue()
