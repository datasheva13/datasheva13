"""Exhaust-sound scoring, one value per second.

What sounds "satisfying" on a parallel twin off-road:
  * revs climbing (spectral centroid rising = pitch going up under load)
  * throttle blips / pops (sharp onsets)
  * sustained loud pulls on climbs (loudness well above the clip's own baseline)
Idle, coasting and stops (quiet + flat) are penalized.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioFeatures:
    loudness: np.ndarray      # dB per second, relative to clip median
    rev_rise: np.ndarray      # positive centroid slope (revs climbing)
    onsets: np.ndarray        # onset strength per second
    score: np.ndarray         # combined 0..1
    onset_times: np.ndarray   # seconds, for cutting on throttle hits


def _robust01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def analyze_audio(wav: str | Path) -> AudioFeatures:
    import librosa

    y, sr = librosa.load(str(wav), sr=None, mono=True)
    n_sec = max(int(len(y) / sr), 1)
    if len(y) < sr:
        z = np.zeros(n_sec)
        return AudioFeatures(z, z, z, z, np.array([]))
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames")
    frames_per_sec = sr / hop

    def per_second(a: np.ndarray, fn=np.mean) -> np.ndarray:
        out = np.zeros(n_sec)
        for s in range(n_sec):
            seg = a[int(s * frames_per_sec): int((s + 1) * frames_per_sec)]
            out[s] = fn(seg) if len(seg) else 0
        return out

    db = 20 * np.log10(np.maximum(per_second(rms), 1e-6))
    loud = db - np.median(db)
    cent = per_second(centroid)
    cent_smooth = np.convolve(cent, np.ones(3) / 3, mode="same")
    rise = np.clip(np.gradient(cent_smooth), 0, None)
    ons = per_second(onset_env, np.max)
    variation = per_second(np.abs(np.diff(rms, prepend=rms[0])), np.mean)

    score = (0.40 * _robust01(loud) + 0.25 * _robust01(rise) + 0.20 * _robust01(ons)
             + 0.15 * _robust01(variation))
    idle = (loud < -6) & (_robust01(variation) < 0.15)
    score[idle] *= 0.3
    return AudioFeatures(loud, rise, ons, np.clip(score, 0, 1),
                         librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop))
