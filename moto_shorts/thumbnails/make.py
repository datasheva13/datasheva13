"""One-word cinematic thumbnails: a strong graded frame, deeper mood, one elegant word.

Frame choice: high scenery score, then pick the frame with the calmest region
(lowest detail) in the upper or middle band, so the word sits on open sky or
soft background, never on busy texture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .. import ffmpeg as ff
from ..edit.grade import Grade, filter_chain

FALLBACK_FONTS = [
    "C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/times.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]


def find_font(preferred: str | Path | None) -> str:
    for p in [preferred, *FALLBACK_FONTS]:
        if p and Path(p).exists():
            return str(p)
    raise FileNotFoundError("No usable font found - set thumbnails.font in config.yaml")


def grab_graded_frames(src: str | Path, times: list[float], out_dir: Path, grade: Grade,
                       size: tuple[int, int] = (3840, 2160), vertical: bool = False) -> list[Path]:
    """Extract full-quality frames with the channel grade (no grain - thumbnails stay clean)."""
    g = Grade(**grade.__dict__)
    g.grain = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    if vertical:
        w, h = 2160, 3840
    crop = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    paths = []
    for i, t in enumerate(times):
        p = out_dir / f"frame_{i}.jpg"
        ff.extract_frame(src, t, p, vf=f"{crop},{filter_chain(g, width=w)}")
        paths.append(p)
    return paths


def calm_spot(img: Image.Image) -> tuple[float, float]:
    """Return (cx, cy) as fractions: the calmest horizontal band for the text."""
    small = img.convert("L").resize((96, 54))
    a = np.asarray(small, dtype=np.float32)
    detail = np.abs(np.diff(a, axis=0))[:, :-1] + np.abs(np.diff(a, axis=1))[:-1, :]
    rows = detail.mean(axis=1)
    bands = {0.22: rows[6:17].mean(), 0.40: rows[16:27].mean(), 0.5: rows[22:32].mean()}
    cy = min(bands, key=bands.get)
    return 0.5, cy


def make_thumbnail(frame: str | Path, word: str, out: str | Path, font_path: str | Path | None = None,
                   letter_spacing: float = 0.35, word_height: float = 0.075,
                   size: tuple[int, int] = (1280, 720)) -> Path:
    img = Image.open(frame).convert("RGB")
    img = img.resize(size if img.width >= img.height else (size[1], size[0]), Image.LANCZOS)
    w, h = img.size

    # extra mood: soft vignette + slightly lifted, cooler blacks
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vig = np.clip(1 - 0.38 * np.clip(r - 0.35, 0, None) ** 1.6, 0.55, 1)[..., None]
    arr = np.asarray(img, dtype=np.float32) / 255
    arr = arr * vig
    arr = 0.03 + arr * 0.97
    arr[..., 2] += 0.008
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    word = word.upper().strip()
    font_px = int(h * word_height / 0.72)
    font = ImageFont.truetype(find_font(font_path), font_px)
    spacing = int(font_px * letter_spacing)
    widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in word]
    total = sum(widths) + spacing * (len(word) - 1)
    max_w = int(w * 0.78)
    if total > max_w:                                     # long word: shrink to fit
        scale = max_w / total
        font = ImageFont.truetype(find_font(font_path), int(font_px * scale))
        spacing = int(spacing * scale)
        widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in word]
        total = sum(widths) + spacing * (len(word) - 1)

    cx, cy = calm_spot(img)
    x0 = int(cx * w - total / 2)
    asc = font.getbbox("H")
    y0 = int(cy * h - (asc[3] - asc[1]) / 2 - asc[1])

    shadow = Image.new("L", img.size, 0)
    text = Image.new("L", img.size, 0)
    ds, dt = ImageDraw.Draw(shadow), ImageDraw.Draw(text)
    x = x0
    for ch, cw in zip(word, widths):
        ds.text((x, y0 + max(2, font.size // 30)), ch, font=font, fill=150)
        dt.text((x, y0), ch, font=font, fill=255)
        x += cw + spacing
    shadow = shadow.filter(ImageFilter.GaussianBlur(font.size / 10))
    img.paste((0, 0, 0), mask=shadow)
    img.paste((244, 241, 236), mask=text)                 # warm off-white, not pure white
    img.save(out, quality=95)
    return Path(out)
