"""Phone handoff for TikTok / Instagram: copies the 1080p video, cover and caption
into a folder your phone syncs (Google Drive / iCloud / OneDrive)."""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from ..captions.generate import caption_text


def handoff(handoff_dir: Path, clip_id: int, social_video: Path, cover: Path | None,
            hook: str, description: str, hashtags: list[str]) -> Path:
    folder = handoff_dir / f"{date.today().isoformat()}_short{clip_id}"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(social_video, folder / "video.mp4")
    if cover and cover.exists():
        shutil.copy2(cover, folder / "cover.jpg")
    (folder / "caption.txt").write_text(caption_text(hook, description, hashtags), encoding="utf-8")
    return folder
