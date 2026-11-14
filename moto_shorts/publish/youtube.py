"""YouTube upload (always private) with title, description, hashtags, thumbnail.

First run opens a browser once to sign in to your channel; the token is saved in
secrets/youtube_token.json. Uploads are private: you flip them to Public yourself.
"""
from __future__ import annotations

from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]


def _service(client_secrets: Path, token_file: Path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds or not creds.valid:
            if not client_secrets.exists():
                raise FileNotFoundError(f"YouTube client secret missing: {client_secrets} (see README step 4)")
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload(cfg, video: Path, title: str, description: str, hashtags: list[str],
           thumbnail: Path | None = None, chapters: list[tuple[float, str]] | None = None) -> str:
    from googleapiclient.http import MediaFileUpload

    ycfg = cfg["youtube"]
    svc = _service(cfg.path("youtube", "client_secrets"), cfg.path("youtube", "token_file"))
    desc = description.strip()
    if chapters and len(chapters) >= 3:
        lines = [f"{int(t // 60)}:{int(t % 60):02d} {name}" for t, name in chapters]
        lines[0] = "0:00 " + chapters[0][1]
        desc += "\n\n" + "\n".join(lines)
    desc += "\n\n" + " ".join(hashtags)
    body = {
        "snippet": {"title": title[:100], "description": desc[:4900],
                    "tags": [h.lstrip("#") for h in hashtags], "categoryId": ycfg.get("category_id", "2")},
        "status": {"privacyStatus": ycfg.get("privacy", "private"), "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(video), chunksize=64 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    req = svc.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"[youtube] uploaded {int(status.progress() * 100)}%")
    vid = resp["id"]
    if thumbnail and thumbnail.exists():
        try:     # custom thumbnails need a phone-verified channel; skip quietly otherwise
            svc.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumbnail))).execute()
        except Exception as e:
            print(f"[youtube] thumbnail not set (verify your channel by phone to enable): {e}")
    return vid
