"""Scenery scoring on the 2 fps proxy, one value per second.

Chin-mount footage always shows the same bars/dash at the bottom, so we score mostly
the upper 65% of the frame (the world ahead) and look for: open vistas (lots of sky,
distant horizon), rich but natural color, good light (golden hour warmth, not flat
midday gray), interesting change (new scenery), and penalize dark/blurry/obstructed frames.

Each second also gets a compact "look" embedding (color layout + histogram) used to
keep consecutive days from repeating the same kind of scenery.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .proxy import PROXY_FPS


@dataclass
class VideoFeatures:
    colorfulness: np.ndarray
    vista: np.ndarray
    light: np.ndarray
    change: np.ndarray
    motion: np.ndarray
    quality: np.ndarray        # 0 = unusable (dark/blurry/obstructed), 1 = fine
    score: np.ndarray          # combined 0..1
    embeddings: np.ndarray     # (n_sec, D) unit vectors


def _robust01(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def frame_features(bgr: np.ndarray, prev_gray: np.ndarray | None) -> tuple[dict, np.ndarray, np.ndarray]:
    import cv2

    h = bgr.shape[0]
    world = bgr[: int(h * 0.65)]
    hsv = cv2.cvtColor(world, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(np.float32), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    b, g, r = [world[..., i].astype(np.float32) for i in range(3)]

    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)

    top = hsv[: int(world.shape[0] * 0.45)]
    sky = ((top[..., 2] > 140) & ((top[..., 1] < 60) | ((top[..., 0] > 90) & (top[..., 0] < 130)))).mean()
    gray = cv2.cvtColor(world, cv2.COLOR_BGR2GRAY)
    edges_top = cv2.Canny(gray[: gray.shape[0] // 2], 60, 160).mean() / 255
    vista = sky * (1 - min(edges_top * 6, 1)) + 0.3 * sky

    bright = val > 0.6
    warmth = float((r[bright].mean() - b[bright].mean()) / 255) if bright.any() else 0.0
    contrast = float(gray.std() / 255)
    light = 0.6 * np.clip(warmth * 4, 0, 1) + 0.4 * np.clip(contrast * 4, 0, 1)

    sharp = cv2.Laplacian(gray, cv2.CV_32F).var()
    mean_v = val.mean()
    # Only darkness and near-zero detail (mud/hand over lens) count as bad. ND motion blur is
    # intentional, so sharpness barely matters: floor 0.7.
    quality = float(np.clip((mean_v - 0.08) / 0.12, 0, 1) * np.clip(sharp / 15, 0.7, 1))

    motion = 0.0
    small = cv2.resize(gray, (80, 45)).astype(np.float32)
    if prev_gray is not None:
        motion = float(np.abs(small - prev_gray).mean() / 255)

    grid = cv2.resize(world, (4, 3), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).reshape(-1)
    hist = hist / max(hist.sum(), 1)
    emb = np.concatenate([grid, hist * 2])
    feats = dict(colorfulness=colorful, vista=vista, light=light, quality=quality, motion=motion,
                 sat=float(sat.mean()), hue_mean=float(hue.mean()))
    return feats, small, emb


def analyze_video(proxy: str | Path) -> VideoFeatures:
    import cv2

    cap = cv2.VideoCapture(str(proxy))
    rows, embs = [], []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f, prev, e = frame_features(frame, prev)
        rows.append(f)
        embs.append(e)
    cap.release()
    if not rows:
        z = np.zeros(1)
        return VideoFeatures(z, z, z, z, z, z, z, np.zeros((1, 60)))

    def per_sec(key: str) -> np.ndarray:
        a = np.array([r[key] for r in rows], dtype=np.float32)
        n = max(len(a) // PROXY_FPS, 1)
        return a[: n * PROXY_FPS].reshape(n, -1).mean(axis=1) if len(a) >= PROXY_FPS else a

    E = np.array(embs, dtype=np.float32)
    n = max(len(E) // PROXY_FPS, 1)
    E = E[: n * PROXY_FPS].reshape(n, PROXY_FPS, -1).mean(axis=1) if len(E) >= PROXY_FPS else E
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)

    # change: distance from the previous ~20 s of look (new scenery = interesting)
    change = np.zeros(len(E))
    for i in range(len(E)):
        past = E[max(0, i - 20): i]
        if len(past):
            change[i] = 1 - float((past @ E[i]).max())

    colorful, vista, light = per_sec("colorfulness"), per_sec("vista"), per_sec("light")
    motion, quality = per_sec("motion"), per_sec("quality")
    m = min(map(len, (colorful, vista, light, motion, quality, change)))
    colorful, vista, light, motion, quality, change = (x[:m] for x in (colorful, vista, light, motion, quality, change))
    E = E[:m]
    # Natural color is good, neon is not: reward colorfulness up to a point
    color_s = _robust01(colorful)
    color_s = np.where(color_s > 0.85, 0.85 - (color_s - 0.85), color_s)
    score = (0.30 * _robust01(vista) + 0.25 * color_s + 0.20 * _robust01(light)
             + 0.15 * _robust01(change) + 0.10 * _robust01(motion))
    score = np.clip(score, 0, 1) * quality
    return VideoFeatures(colorful, vista, light, change, motion, quality, score, E)
