"""Review dashboard: http://127.0.0.1:8765

Review each short (hook, thumbnail word, trim, grade), Approve -> POSTED (kept 7 days),
Reject -> deleted and never suggested again. Upload your own edits for grading/export.
"""
from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .. import ffmpeg as ff
from ..captions.generate import caption_text
from ..config import ROOT, load_config
from ..edit.grade import filter_chain
from ..pipeline import POSTED_KEEP_DAYS, Pipeline, _age_days

cfg = load_config()
pipe = Pipeline(cfg)
app = FastAPI(title="Moto Shorts")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_lock = threading.Lock()
_jobs: dict[int, str] = {}


def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(cfg["paths"].get("timezone", "America/Denver"))
    except Exception:
        return timezone.utc


def local_time(iso: str | None) -> str:
    """'2026-11-17T13:15:00+00:00' -> 'Nov 17, 6:15 AM MST' in your time zone."""
    if not iso:
        return ""
    t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    t = t.astimezone(_tz())
    return t.strftime("%b %d, %I:%M %p %Z").replace(" 0", " ")


def mmss(sec: float | None) -> str:
    sec = float(sec or 0)
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


templates.env.filters["local"] = local_time
templates.env.filters["mmss"] = mmss


def _run(clip_id: int, label: str, fn, *args) -> None:
    _jobs[clip_id] = label
    try:
        with _lock:                          # one heavy job at a time (GPU/CPU)
            fn(*args)
    finally:
        _jobs.pop(clip_id, None)


def _rerender(bg: BackgroundTasks, clip_id: int, label: str) -> None:
    clip = pipe.db.clip(clip_id)
    fn = pipe.render_short_clip if clip["kind"] == "short" else pipe.render_long_clip
    _jobs[clip_id] = label + " (queued)"
    bg.add_task(_run, clip_id, label, fn, clip_id)


def _view(r) -> dict:
    d = dict(r)
    d.pop("embedding", None)
    for k in ("captions", "hashtags", "thumb_options", "segments"):
        d[k] = json.loads(d[k]) if d.get(k) else ([] if k != "captions" else {})
    d["job"] = _jobs.get(d["id"])
    d["caption_text"] = caption_text(d.get("hook") or "", d.get("description") or "", d["hashtags"])
    d["yt_description"] = ((d.get("description") or "") + "\n\n" + " ".join(d["hashtags"])).strip()
    hooks = list(d["captions"].get("hooks", []) if isinstance(d["captions"], dict) else [])
    if d.get("hook") and d["hook"] not in hooks:
        hooks.append(d["hook"])                     # your own typed hook stays selectable
    d["hook_options"] = hooks
    d["days_left"] = max(0, round(POSTED_KEEP_DAYS - _age_days(d.get("posted_at")), 1)) if d.get("posted_at") else None
    src = pipe.db.source(d["source_id"]) if d.get("source_id") else None
    d["source_name"] = src["filename"] if src else ""
    d["source_duration"] = float(src["duration"] or 0) if src else 0
    return d


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    review = [_view(r) for r in pipe.db.clips(("rendered", "pending", "failed"))]
    posted = [_view(r) for r in pipe.db.clips(("posted",))]
    return templates.TemplateResponse(request, "index.html", {
        "review": review, "posted": posted, "status": pipe.status(), "jobs": _jobs,
        "handoff_dir": str(cfg.path("paths", "handoff_dir")), "keep_days": POSTED_KEEP_DAYS,
        "yt_mode": cfg["youtube"].get("mode", "manual"), "now": local_time(datetime.now(timezone.utc).isoformat())})


@app.get("/media/{clip_id}/{what}")
def media(clip_id: int, what: str):
    clip = pipe.db.clip(clip_id)
    if not clip:
        raise HTTPException(404)
    if what == "preview":
        p = clip["social_path"] or clip["master_path"]
    elif what == "instagram":
        p = clip["instagram_path"]
    elif what == "thumb":
        p = clip["thumb_path"]
    elif what.startswith("frame"):
        opts = json.loads(clip["thumb_options"] or "[]")
        idx = int(what[5:] or 0)
        p = opts[idx] if idx < len(opts) else None
    else:
        raise HTTPException(404)
    if not p or not Path(p).exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.get("/compare/{clip_id}/{which}.jpg")
def compare(clip_id: int, which: str, strength: float = 1.0):
    """Before/after frames for the grade slider."""
    clip = pipe.db.clip(clip_id)
    if not clip or clip["kind"] != "short" or not clip["master_path"]:
        raise HTTPException(404)
    out_dir = Path(clip["master_path"]).parent / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = pipe.ensure_local(clip["source_id"])
    t = clip["start"] + clip["duration"] * 0.4
    tag = f"{clip['start']:.1f}"
    if which == "before":
        out = out_dir / f"before_{tag}.jpg"
        if not out.exists():
            ff.extract_frame(src, t, out, vf="scale=1280:-2")
    else:
        out = out_dir / f"after_{tag}_{int(strength * 100)}.jpg"
        if not out.exists():
            g = pipe.grade.with_strength(strength)
            g.grain = 0
            ff.extract_frame(src, t, out, vf=f"scale=1280:-2,{filter_chain(g, width=1280)}")
    return FileResponse(out)


@app.post("/clip/{clip_id}/save")
def save(clip_id: int, title: str = Form(""), description: str = Form(""), hashtags: str = Form("")):
    tags = [t if t.startswith("#") else f"#{t}" for t in hashtags.replace(",", " ").split() if t.strip()][:3]
    pipe.db.update_clip(clip_id, title=title.strip(), description=description.strip(), hashtags=tags)
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/hook")
def set_hook(clip_id: int, bg: BackgroundTasks, hook: str = Form(""), custom: str = Form("")):
    """Pick a suggested hook or type your own -> saved and burned into the video (re-render)."""
    text = (custom or hook).strip()
    if text:
        pipe.db.update_clip(clip_id, hook=text)
        _rerender(bg, clip_id, "putting the new hook on the video")
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/thumbnail")
def thumbnail(clip_id: int, pick: str = Form(""), word: str = Form(""), frame: int = Form(1)):
    chosen = (pick or word).strip()
    clip = pipe.db.clip(clip_id)
    chosen = chosen.split()[0].upper() if chosen else (clip["thumb_word"] or "RIDE")
    pipe.rethumb(clip_id, chosen, frame)
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/trim")
def trim(clip_id: int, bg: BackgroundTasks, start_delta: float = Form(0.0), end_delta: float = Form(0.0)):
    if start_delta or end_delta:
        pipe.trim(clip_id, start_delta, end_delta)
        _rerender(bg, clip_id, "trimming")
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/rerender")
def rerender(clip_id: int, bg: BackgroundTasks, strength: float = Form(1.0), music: str = Form("")):
    pipe.db.update_clip(clip_id, grade_strength=max(0.0, min(1.0, strength)), music="1" if music else None)
    _rerender(bg, clip_id, "re-rendering")
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/new-captions")
def new_captions(clip_id: int, bg: BackgroundTasks):
    pipe.db.update_clip(clip_id, captions=None)
    _rerender(bg, clip_id, "writing new captions")
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/approve")
def approve(clip_id: int, bg: BackgroundTasks):
    def job():
        res = pipe.approve(clip_id)
        if "youtube_error" in res:
            pipe.db.update_clip(clip_id, error="YouTube upload failed: " + res["youtube_error"])
    bg.add_task(_run, clip_id, "preparing posts", job)
    return RedirectResponse("/#posted", status_code=303)


@app.post("/clip/{clip_id}/reject")
def reject(clip_id: int):
    pipe.reject(clip_id)
    return RedirectResponse("/", status_code=303)


@app.post("/upload/{kind}")
def upload(kind: str, bg: BackgroundTasks, file: UploadFile = File(...)):
    """Your own edited clip. short: blur layout + grade + 3 platform exports. long: grade only."""
    if kind not in ("short", "long"):
        raise HTTPException(400)
    dest_dir = cfg.sub("uploads")
    name = Path(file.filename or "upload.mp4").name
    dest = dest_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh, length=16 * 1024 * 1024)
    bg.add_task(_run, 0, f"processing your {kind} upload", pipe.add_upload, dest, kind)
    return RedirectResponse("/", status_code=303)


@app.post("/run-now")
def run_now(bg: BackgroundTasks):
    bg.add_task(_run, 0, "daily run", pipe.daily)
    return RedirectResponse("/", status_code=303)


@app.post("/build-long")
def build_long(bg: BackgroundTasks):
    bg.add_task(_run, 0, "building long video", pipe.maybe_build_long, True)
    return RedirectResponse("/", status_code=303)


@app.post("/gopro-token")
def gopro_token(token: str = Form(...)):
    """Save a fresh GoPro token into .env (local file, never committed)."""
    env = ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    lines = [l for l in lines if not l.startswith("GOPRO_ACCESS_TOKEN=")] + [f"GOPRO_ACCESS_TOKEN={token.strip()}"]
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import os
    os.environ["GOPRO_ACCESS_TOKEN"] = token.strip()
    pipe.set_status(gopro_ok=None, gopro_error=None)
    return RedirectResponse("/", status_code=303)
