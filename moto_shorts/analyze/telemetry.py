"""Minimal GoPro GPMF parser (HERO11): GPS speed + altitude per second.

GPMF is KLV: 4-byte key, 1-byte type, 1-byte struct size, 2-byte repeat count,
payload padded to 4 bytes. GPS5 rows are (lat, lon, alt, speed2d, speed3d) as int32
divided by SCAL. We only need speed and altitude to find climbs and fast flowing sections.
"""
from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import ffmpeg as ff

_TYPES = {"b": ("b", 1), "B": ("B", 1), "s": ("h", 2), "S": ("H", 2), "l": ("i", 4), "L": ("I", 4),
          "f": ("f", 4), "d": ("d", 8), "j": ("q", 8), "J": ("Q", 8)}


@dataclass
class Telemetry:
    speed_kmh: np.ndarray      # per second
    altitude_m: np.ndarray     # per second
    climb: np.ndarray          # m/s vertical (positive = climbing)

    @property
    def empty(self) -> bool:
        return len(self.speed_kmh) == 0


def extract_gpmf(src: str | Path) -> bytes:
    info = ff.probe(src)
    if not info.data_streams:
        return b""
    idx = info.data_streams[0]
    proc = subprocess.run([ff.FFMPEG, "-v", "error", "-i", str(src), "-map", f"0:{idx}", "-codec", "copy",
                           "-f", "rawvideo", "-"], capture_output=True)
    return proc.stdout


def parse_gps5(data: bytes) -> list[np.ndarray]:
    """Return a list of GPS5 blocks (each ~1 s of samples, already scaled)."""
    blocks: list[np.ndarray] = []

    def walk(buf: bytes, scal: list[float] | None = None) -> None:
        pos = 0
        local_scal = scal
        while pos + 8 <= len(buf):
            key = buf[pos:pos + 4].decode("latin-1")
            typ = chr(buf[pos + 4])
            size = buf[pos + 5]
            repeat = struct.unpack(">H", buf[pos + 6:pos + 8])[0]
            length = size * repeat
            payload = buf[pos + 8:pos + 8 + length]
            pos += 8 + ((length + 3) & ~3)
            if typ == "\x00":                       # nested container (DEVC, STRM)
                walk(payload, None if key == "STRM" else local_scal)
                continue
            if key == "SCAL" and typ in _TYPES:
                fmt, n = _TYPES[typ]
                count = length // n
                local_scal = list(struct.unpack(">" + fmt * count, payload[: count * n]))
            elif key == "GPS5" and typ == "l" and size == 20:
                arr = np.frombuffer(payload, dtype=">i4").reshape(repeat, 5).astype(np.float64)
                sc = np.array(local_scal if local_scal and len(local_scal) == 5 else [1e7, 1e7, 1e3, 1e3, 1e2])
                blocks.append(arr / sc)

    try:
        walk(data)
    except (struct.error, ValueError, UnicodeDecodeError):
        pass
    return blocks


def read_telemetry(src: str | Path) -> Telemetry:
    blocks = parse_gps5(extract_gpmf(src))
    if not blocks:
        e = np.array([])
        return Telemetry(e, e, e)
    speed = np.array([b[:, 3].mean() * 3.6 for b in blocks])
    alt = np.array([np.median(b[:, 2]) for b in blocks])
    alt_s = np.convolve(alt, np.ones(5) / 5, mode="same") if len(alt) >= 5 else alt
    climb = np.gradient(alt_s) if len(alt_s) > 1 else np.zeros_like(alt_s)
    return Telemetry(speed, alt, climb)


def telemetry_score(t: Telemetry, n_sec: int) -> np.ndarray:
    """0..1 per second: flowing speed (not max speed) + climbing. Zero-filled if no GPS."""
    out = np.zeros(n_sec)
    if t.empty:
        return out
    m = min(n_sec, len(t.speed_kmh))
    flow = np.clip(t.speed_kmh[:m] / 45, 0, 1)            # 45 km/h on a trail = lively
    flow = np.where(t.speed_kmh[:m] < 4, 0, flow)          # stopped
    climb = np.clip(t.climb[:m] / 1.5, 0, 1)
    out[:m] = 0.6 * flow + 0.4 * climb
    return out
