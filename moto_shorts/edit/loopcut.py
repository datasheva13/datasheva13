"""Find the smoothest place for the loop's internal cut (clip end -> clip start).

The short plays [split -> end] then [start -> split]. The opening (split) is fixed:
attention comes first. Only `start` and `end` are nudged (+-1.5 s) to the pair whose
sound (engine pitch + loudness) and picture (colour/layout, little motion) match best,
so the jump from end back to start is hard to notice.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .. import ffmpeg as ff

SR = 16000
HOP = 320            # 20 ms
FPS = 10
W, H = 96, 54


def _audio(src: Path, t0: float, dur: float) -> np.ndarray:
    proc = subprocess.run([ff.FFMPEG, "-v", "error", "-ss", f"{max(t0, 0):.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                           "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(proc.stdout, np.float32)


def _frames(src: Path, t0: float, dur: float) -> np.ndarray:
    proc = subprocess.run([ff.FFMPEG, "-v", "error", "-ss", f"{max(t0, 0):.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                           "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                          capture_output=True)
    n = len(proc.stdout) // (W * H * 3)
    return np.frombuffer(proc.stdout[: n * W * H * 3], np.uint8).reshape(n, H, W, 3).astype(np.float32) / 255


def _audio_features(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per 20 ms: loudness (dB) and brightness/pitch proxy (spectral centroid, Hz)."""
    n = max(len(y) // HOP - 1, 0)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    frames = np.stack([y[i * HOP: i * HOP + 2 * HOP] for i in range(n)]) * np.hanning(2 * HOP)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(2 * HOP, 1 / SR)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)
    centroid = (spec * freqs).sum(axis=1) / (spec.sum(axis=1) + 1e-9)
    return rms_db, centroid


def best_loop_pair(src: str | Path, start: float, end: float, split: float, search: float = 1.5,
                   src_duration: float | None = None, min_part: float = 2.0) -> tuple[float, float]:
    """Return (start, end) nudged for the smoothest end->start cut. Falls back to the input."""
    src = Path(src)
    s0, e0 = max(start - search, 0.0), end - search
    span = 2 * search
    if src_duration:
        span_e = min(span, src_duration - e0)
    else:
        span_e = span
    try:
        a_s, a_e = _audio(src, s0, span), _audio(src, e0, span_e)
        f_s, f_e = _frames(src, s0, span), _frames(src, e0, span_e)
    except Exception:
        return start, end
    if len(f_s) < 3 or len(f_e) < 3 or len(a_s) < SR * 0.3 or len(a_e) < SR * 0.3:
        return start, end
    rms_s, cen_s = _audio_features(a_s)
    rms_e, cen_e = _audio_features(a_e)
    motion_s = np.r_[0, np.abs(np.diff(f_s, axis=0)).mean(axis=(1, 2, 3))]
    motion_e = np.r_[0, np.abs(np.diff(f_e, axis=0)).mean(axis=(1, 2, 3))]

    def at(arr: np.ndarray, t: float, rate: float) -> float:
        i = int(np.clip(round(t * rate), 0, len(arr) - 1))
        return float(arr[i])

    best, best_cost = (start, end), np.inf
    for ts in np.arange(0, span + 1e-6, 0.1):            # candidate starts
        s = s0 + ts
        if s > split - min_part:
            break
        fi_s = int(np.clip(round(ts * FPS), 0, len(f_s) - 1))
        for te in np.arange(0, span_e + 1e-6, 0.1):     # candidate ends
            e = e0 + te
            if e < split + min_part:
                continue
            fi_e = int(np.clip(round(te * FPS), 0, len(f_e) - 1))
            look = float(np.abs(f_e[fi_e] - f_s[fi_s]).mean())                     # 0..1
            loud = abs(at(rms_e, te, SR / HOP) - at(rms_s, ts, SR / HOP)) / 12      # 12 dB = very different
            pitch = abs(at(cen_e, te, SR / HOP) - at(cen_s, ts, SR / HOP)) / 1500
            motion = (motion_e[fi_e] + motion_s[fi_s]) * 4
            drift = (abs(s - start) + abs(e - end)) * 0.03                          # stay close to the plan
            cost = 0.40 * look * 4 + 0.35 * min(loud + pitch, 2) + 0.15 * motion + 0.10 * drift
            if cost < best_cost:
                best_cost, best = cost, (round(s, 2), round(e, 2))
    return best
