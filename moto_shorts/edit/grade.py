"""Cinematic-but-natural color grade for GoPro Flat footage, as an FFmpeg filter chain.

Order: per-clip normalization -> Flat-to-709 contrast -> tone curve -> restrained color
-> green taming -> optional LUT -> grain -> vignette.
Every creative value is blended toward neutral by `strength` (0..1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Grade:
    strength: float = 1.0
    base_contrast: float = 1.10
    base_saturation: float = 1.12
    curve_master: list = field(default_factory=lambda: [[0, 0.025], [0.25, 0.2], [0.5, 0.5], [0.75, 0.79], [1, 0.955]])
    shadows_rgb: list = field(default_factory=lambda: [-0.02, 0.0, 0.035])
    midtones_rgb: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    highlights_rgb: list = field(default_factory=lambda: [0.03, 0.01, -0.025])
    saturation: float = 0.94
    greens_desat: float = 0.12
    grain: float = 6
    vignette: float = 0.32
    lut: str | None = None
    lut_mix: float = 1.0
    normalize_exposure: bool = True
    normalize_white_balance: bool = True
    target_luma: float = 0.43
    max_exposure_ev: float = 0.6
    name: str = "channel_look"

    @classmethod
    def load(cls, path: str | Path) -> "Grade":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.__dict__, sort_keys=False), encoding="utf-8")

    def with_strength(self, s: float) -> "Grade":
        g = Grade(**self.__dict__)
        g.strength = max(0.0, min(1.0, s))
        return g


@dataclass
class ClipNormalization:
    """Per-clip correction measured from sample frames (see measure_normalization)."""
    exposure_ev: float = 0.0
    gains_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


def measure_normalization(frames_rgb: list[np.ndarray], grade: Grade) -> ClipNormalization:
    """Gray-world white balance on midtones + exposure nudge toward target luma.

    frames_rgb: float arrays in 0..1, shape (H, W, 3). Small proxies are fine.
    Corrections are deliberately gentle so the scene's natural mood survives
    (a sunset should stay warm, a forest should stay dark-ish).
    """
    if not frames_rgb:
        return ClipNormalization()
    px = np.concatenate([f.reshape(-1, 3) for f in frames_rgb])
    luma = px @ np.array([0.2126, 0.7152, 0.0722])
    norm = ClipNormalization()
    if grade.normalize_white_balance:
        mid = px[(luma > 0.2) & (luma < 0.8)]
        if len(mid) > 100:
            means = mid.mean(axis=0)
            gray = means.mean()
            gains = gray / np.maximum(means, 1e-4)
            gains = 1.0 + (gains - 1.0) * 0.5          # only correct half the cast
            gains = np.clip(gains, 0.92, 1.08)
            norm.gains_rgb = tuple(float(g) for g in gains)
    if grade.normalize_exposure:
        mean_luma = float(np.median(luma))
        if mean_luma > 1e-3:
            ev = math.log2(grade.target_luma / mean_luma) * 0.6
            norm.exposure_ev = float(np.clip(ev, -grade.max_exposure_ev, grade.max_exposure_ev))
    return norm


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _curve_str(points: list, strength: float) -> str:
    pts = []
    for x, y in points:
        y2 = _lerp(float(x), float(y), strength)
        pts.append(f"{float(x):.4f}/{min(max(y2, 0.0), 1.0):.4f}")
    return " ".join(pts)


def filter_chain(grade: Grade, norm: ClipNormalization | None = None, width: int | None = None) -> str:
    """Return a comma-joined FFmpeg -vf chain (no input/output labels)."""
    s = grade.strength
    norm = norm or ClipNormalization()
    f: list[str] = []

    # 1. per-clip normalization (always full strength: it's correction, not style)
    r, g, b = norm.gains_rgb
    if any(abs(x - 1.0) > 0.003 for x in (r, g, b)):
        f.append(f"colorchannelmixer=rr={r:.4f}:gg={g:.4f}:bb={b:.4f}")
    if abs(norm.exposure_ev) > 0.01:
        f.append(f"exposure=exposure={norm.exposure_ev:.3f}")

    # 2. Flat -> 709 base
    f.append(f"eq=contrast={_lerp(1, grade.base_contrast, s):.4f}:saturation={_lerp(1, grade.base_saturation, s):.4f}")

    # 3. tone curve
    f.append(f"curves=master='{_curve_str(grade.curve_master, s)}'")

    # 4. restrained color balance
    sh, md, hl = grade.shadows_rgb, grade.midtones_rgb, grade.highlights_rgb
    vals = [v * s for v in (*sh, *md, *hl)]
    if any(abs(v) > 1e-4 for v in vals):
        f.append("colorbalance=rs={:.4f}:gs={:.4f}:bs={:.4f}:rm={:.4f}:gm={:.4f}:bm={:.4f}:"
                 "rh={:.4f}:gh={:.4f}:bh={:.4f}:pl=1".format(*vals))
    if grade.greens_desat > 0:
        f.append(f"huesaturation=saturation={-grade.greens_desat * s:.4f}:colors=g+y:strength=1")
    f.append(f"eq=saturation={_lerp(1, grade.saturation, s):.4f}")

    # 5. optional LUT
    if grade.lut and Path(grade.lut).exists():
        lut_path = str(Path(grade.lut).resolve()).replace("\\", "/").replace(":", "\\:")
        f.append(f"lut3d=file='{lut_path}'")

    # 6. film grain (temporal, luma-weighted) - scaled with resolution so it reads the same
    if grade.grain > 0 and s > 0:
        scale = (width or 3840) / 3840
        amount = max(1, round(grade.grain * s * max(scale, 0.35)))
        f.append(f"noise=c0s={amount}:c0f=t+u")

    # 7. vignette (angle: PI/5 is very subtle, PI/3 strong)
    if grade.vignette > 0 and s > 0:
        angle = _lerp(0.05, math.pi / 3, grade.vignette * s)
        f.append(f"vignette=angle={angle:.4f}:mode=forward")
    return ",".join(f)


def write_cube_lut(grade: Grade, path: str | Path, size: int = 33) -> Path:
    """Bake the tone curve + base contrast/saturation into a .cube LUT (for use in
    DaVinci/Premiere or as `lut:`). Color balance/grain/vignette are left out on purpose."""
    xs = np.array([p[0] for p in grade.curve_master], float)
    ys = np.array([_lerp(p[0], p[1], grade.strength) for p in grade.curve_master], float)
    lines = [f'TITLE "{grade.name}"', f"LUT_3D_SIZE {size}"]
    grid = np.linspace(0, 1, size)
    c = _lerp(1, grade.base_contrast, grade.strength)
    sat = _lerp(1, grade.base_saturation, grade.strength) * _lerp(1, grade.saturation, grade.strength)
    for bv in grid:
        for gv in grid:
            for rv in grid:
                rgb = np.array([rv, gv, bv])
                rgb = (rgb - 0.5) * c + 0.5
                rgb = np.interp(np.clip(rgb, 0, 1), xs, ys)
                y = rgb @ np.array([0.2126, 0.7152, 0.0722])
                rgb = np.clip(y + (rgb - y) * sat, 0, 1)
                lines.append(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)
