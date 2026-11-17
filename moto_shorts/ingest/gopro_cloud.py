"""GoPro Cloud (Quik) ingest via the *unofficial* API the GoPro web media library uses.

Auth: the `gp_access_token` cookie from https://gopro.com/media-library (see README).
It expires periodically; when it does we raise TokenExpired, the dashboard shows a
banner, and the local inbox folder keeps working as a fallback.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

API = "https://api.gopro.com"
ACCEPT = "application/vnd.gopro.jk.media+json; version=2.0.0"
FIELDS = "id,captured_at,content_title,filename,file_size,type,source_duration,width,height,camera_model,ready_to_view"
VIDEO_EXT = {".mp4", ".mov", ".MP4", ".MOV"}


class TokenExpired(RuntimeError):
    pass


class GoProCloud:
    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self.token = token or os.environ.get("GOPRO_ACCESS_TOKEN", "")
        if not self.token:
            raise TokenExpired("No GOPRO_ACCESS_TOKEN set")
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.token}", "Accept": ACCEPT,
                               "User-Agent": "moto-shorts/1.0"})

    def _get(self, path: str, **params) -> dict:
        r = self.s.get(API + path, params=params, timeout=60)
        if r.status_code in (401, 403):
            raise TokenExpired("GoPro token rejected - paste a fresh one in the dashboard")
        r.raise_for_status()
        return r.json()

    def list_videos(self, newer_than_days: int = 0, max_pages: int = 40) -> list[dict]:
        """All ready videos in the library (newer_than_days <= 0 means no age limit)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=newer_than_days)) if newer_than_days > 0 else None
        out: list[dict] = []
        for page in range(1, max_pages + 1):
            data = self._get("/media/search", fields=FIELDS, order_by="captured_at", per_page=50,
                             page=page, type="Video", processing_states="ready")
            items = data.get("_embedded", {}).get("media", [])
            for m in items:
                cap = m.get("captured_at")
                if cutoff and cap and datetime.fromisoformat(cap.replace("Z", "+00:00")) < cutoff:
                    return out
                if m.get("type") == "Video" and m.get("ready_to_view", "ready") == "ready":
                    out.append(m)
            pages = data.get("_pages", {})
            if not items or page >= int(pages.get("total_pages", page)):
                break
        return out

    def source_url(self, media_id: str) -> tuple[str, str]:
        """URL of the original full-quality file (not a proxy)."""
        data = self._get(f"/media/{media_id}/download")
        emb = data.get("_embedded", {})
        files = [f for f in emb.get("files", []) if f.get("url")]
        if files:
            best = max(files, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))
            return best["url"], data.get("filename") or f"{media_id}.mp4"
        for v in emb.get("variations", []):
            if v.get("label") == "source" and v.get("url"):
                return v["url"], data.get("filename") or f"{media_id}.mp4"
        raise RuntimeError(f"No downloadable source for media {media_id}")

    def download(self, media_id: str, dest_dir: Path) -> Path:
        url, filename = self.source_url(media_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{media_id}_{Path(filename).name}"
        if dest.exists():
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, stream=True, timeout=120) as r:      # pre-signed URL: no auth header
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    fh.write(chunk)
        tmp.rename(dest)
        return dest


def inbox_files(inbox: Path) -> list[Path]:
    """Fallback: any video dropped into the inbox folder (e.g. exported from Quik or the SD card)."""
    inbox.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in inbox.iterdir() if p.suffix in VIDEO_EXT and not p.name.startswith("."))


def move_into(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src.resolve() != dest.resolve():
        shutil.move(str(src), dest)
    return dest
