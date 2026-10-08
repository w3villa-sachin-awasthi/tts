"""maya-research/Veena TTS — batch WAV + chunked SNAC decode for SSE streaming."""

from __future__ import annotations

import io
import os
import threading
from collections.abc import Iterator
from queue import Empty, Queue

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

SPEAKERS = [
    ("kavya", "Kavya (female)"),
    ("agastya", "Agastya (male)"),
    ("maitri", "Maitri (female)"),
    ("vinaya", "Vinaya (female)"),
]
SAMPLE_RATE = 24000

START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266
SNAC_FRAME = 7


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


class _TokenStreamer(BaseStreamer):
    """Yield generated token ids (skip prompt)."""

    def __init__(self) -> None:
        self.queue: Queue[int | None] = Queue()
        self._skip_prompt = True

    def put(self, value) -> None:
        if len(value.shape) > 1:
            value = value[0]
        ids = value.tolist()
        if self._skip_prompt:
            self._skip_prompt = False
            return
        for token_id in ids:
            self.queue.put(int(token_id))

    def end(self) -> None:
        self.queue.put(None)


class VeenaEngine:
    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self.model_id = _env("VEENA_MODEL_ID", "maya-research/Veena")
        self._loaded = False
        self.model = None
        self.tokenizer = None
        self.snac = None
        self._stream_frames = max(1, int(_env("VEENA_STREAM_FRAMES", "4") or "4"))

    def load(self) -> None:
        if self._loaded:
            return
        with self.lock:
            if self._loaded:
                return
            from snac import SNAC

            load_4bit = _env("VEENA_LOAD_IN_4BIT", "0").lower() in ("1", "true", "yes")
            kwargs: dict = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            if load_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                kwargs["torch_dtype"] = dtype

            print(f"Loading Veena {self.model_id} on {self.device}…", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            if self.device != "cuda" or kwargs.get("device_map") is None:
                self.model.to(self.device)
            self.model.eval()

            snac_device = self.device if self.device in ("cuda", "cpu") else "cpu"
            self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            self.snac.to(snac_device)
            self._loaded = True

    def _input_ids(self, text: str, speaker: str) -> torch.Tensor:
        prompt = f"<spk_{speaker}> {text}"
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [
            START_OF_HUMAN_TOKEN,
            *prompt_tokens,
            END_OF_HUMAN_TOKEN,
            START_OF_AI_TOKEN,
            START_OF_SPEECH_TOKEN,
        ]
        device = next(self.model.parameters()).device
        return torch.tensor([tokens], device=device)

    def _max_new_tokens(self, text: str) -> int:
        return min(int(len(text) * 1.3) * 7 + 21, 700)

    def _is_snac(self, token_id: int) -> bool:
        return AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)

    def _decode_frames(self, snac_tokens: list[int]) -> np.ndarray | None:
        if not snac_tokens or len(snac_tokens) % SNAC_FRAME != 0:
            return None
        snac_device = next(self.snac.parameters()).device
        codes_lvl: list[list[int]] = [[], [], []]
        offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]
        for i in range(0, len(snac_tokens), SNAC_FRAME):
            codes_lvl[0].append(snac_tokens[i] - offsets[0])
            codes_lvl[1].append(snac_tokens[i + 1] - offsets[1])
            codes_lvl[1].append(snac_tokens[i + 4] - offsets[4])
            codes_lvl[2].append(snac_tokens[i + 2] - offsets[2])
            codes_lvl[2].append(snac_tokens[i + 3] - offsets[3])
            codes_lvl[2].append(snac_tokens[i + 5] - offsets[5])
            codes_lvl[2].append(snac_tokens[i + 6] - offsets[6])
        hierarchical = []
        for lvl in codes_lvl:
            tensor = torch.tensor(lvl, dtype=torch.int32, device=snac_device).unsqueeze(0)
            if torch.any((tensor < 0) | (tensor > 4095)):
                raise ValueError("Invalid SNAC token values")
            hierarchical.append(tensor)
        with torch.no_grad():
            audio = self.snac.decode(hierarchical)
        return audio.squeeze().clamp(-1, 1).detach().cpu().numpy().astype(np.float32)

    def iter_synthesize(self, text: str, speaker: str) -> Iterator[np.ndarray]:
        if speaker not in {s for s, _ in SPEAKERS}:
            raise ValueError("Unknown Veena speaker.")
        self.load()
        input_ids = self._input_ids(text, speaker)
        streamer = _TokenStreamer()
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self._max_new_tokens(text),
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
            streamer=streamer,
        )
        error: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)
            finally:
                streamer.end()

        with self.lock:
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            buf: list[int] = []
            pending_frames: list[int] = []
            while True:
                try:
                    token = streamer.queue.get(timeout=0.5)
                except Empty:
                    if error:
                        raise error[0]
                    if not thread.is_alive() and streamer.queue.empty():
                        break
                    continue
                if token is None:
                    break
                if not self._is_snac(token):
                    continue
                pending_frames.append(token)
                if len(pending_frames) < SNAC_FRAME:
                    continue
                frame = pending_frames[:SNAC_FRAME]
                pending_frames = pending_frames[SNAC_FRAME:]
                buf.extend(frame)
                if len(buf) // SNAC_FRAME < self._stream_frames:
                    continue
                take = self._stream_frames * SNAC_FRAME
                chunk_tokens = buf[:take]
                buf = buf[take:]
                audio = self._decode_frames(chunk_tokens)
                if audio is not None and audio.size:
                    yield audio
            if buf:
                usable = (len(buf) // SNAC_FRAME) * SNAC_FRAME
                if usable:
                    audio = self._decode_frames(buf[:usable])
                    if audio is not None and audio.size:
                        yield audio
            thread.join(timeout=5)
            if error:
                raise error[0]

    def synthesize(self, text: str, speaker: str) -> bytes:
        chunks = list(self.iter_synthesize(text, speaker))
        if not chunks:
            raise RuntimeError("Veena returned no audio.")
        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(chunks), SAMPLE_RATE, format="WAV")
        return buffer.getvalue()
