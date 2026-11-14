"""Phone handoff for TikTok / Instagram: copies the 1080p videos, covers and caption
into a folder your phone syncs (Google Drive / iCloud / OneDrive).

  tiktok.mp4        vertical, blurred background
  instagram.mp4     horizontal frame turned sideways (viewer rotates the phone)
  cover.jpg         16:9 thumbnail (YouTube)
  cover_instagram.jpg  same thumbnail rotated to match instagram.mp4
  caption.txt       hook + description + 3 hashtags
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from PIL import Image

from ..captions.generate import caption_text


def handoff(handoff_dir: Path, clip_id: int, social_video: Path, cover: Path | None,
            hook: str, description: str, hashtags: list[str],
            instagram_video: Path | None = None, rotate: str = "clockwise") -> Path:
    folder = handoff_dir / f"{date.today().isoformat()}_short{clip_id}"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(social_video, folder / "tiktok.mp4")
    ig = instagram_video if instagram_video and instagram_video.exists() else social_video
    shutil.copy2(ig, folder / "instagram.mp4")
    if cover and cover.exists():
        shutil.copy2(cover, folder / "cover.jpg")
        if ig != social_video:
            img = Image.open(cover)
            img.transpose(Image.Transpose.ROTATE_270 if rotate == "clockwise" else Image.Transpose.ROTATE_90) \
               .save(folder / "cover_instagram.jpg", quality=95)
    (folder / "caption.txt").write_text(caption_text(hook, description, hashtags), encoding="utf-8")
    return folder
