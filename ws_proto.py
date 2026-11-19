"""Minimal WebSocket helpers for stdlib HTTP handlers / asyncio clients."""

from __future__ import annotations

import base64
import hashlib
import os
import struct
from typing import BinaryIO

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def accept_key(sec_key: str) -> str:
    return base64.b64encode(hashlib.sha1((sec_key + _GUID).encode()).digest()).decode()


def recv_text(rfile: BinaryIO) -> str | None:
    hdr = rfile.read(2)
    if not hdr or len(hdr) < 2:
        return None
    opcode, b2 = hdr[0] & 0x0F, hdr[1]
    masked, length = (b2 >> 7) & 1, b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", rfile.read(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", rfile.read(8))[0]
    mask = rfile.read(4) if masked else b""
    data = bytearray(rfile.read(length))
    if masked:
        for i in range(len(data)):
            data[i] ^= mask[i % 4]
    if opcode == 8:
        return None
    if opcode not in (1, 2):
        return ""
    return data.decode("utf-8", errors="replace")


def send_text(wfile: BinaryIO, text: str, *, mask: bool = False) -> None:
    raw = text.encode("utf-8")
    header = bytearray([0x81])
    n = len(raw)
    if n < 126:
        header.append((0x80 if mask else 0) | n)
    elif n < 65536:
        header.append((0x80 if mask else 0) | 126)
        header.extend(struct.pack("!H", n))
    else:
        header.append((0x80 if mask else 0) | 127)
        header.extend(struct.pack("!Q", n))
    if mask:
        key = os.urandom(4)
        header.extend(key)
        raw = bytes(b ^ key[i % 4] for i, b in enumerate(raw))
    wfile.write(header + raw)
    wfile.flush()
