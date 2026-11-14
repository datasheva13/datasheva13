"""Review dashboard: http://127.0.0.1:8765

Preview each short, pick a hook, edit title/description/hashtags, choose the thumbnail
word, adjust grade strength (before/after slider), then Approve: YouTube private upload
+ phone package for TikTok/Instagram with a copy-ready caption.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ..captions.generate import caption_text
from ..config import ROOT, load_config
from ..edit.grade import filter_chain
from .. import ffmpeg as ff
from ..pipeline import Pipeline

cfg = load_config()
pipe = Pipeline(cfg)
app = FastAPI(title="Moto Shorts")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_lock = threading.Lock()
_jobs: dict[int, str] = {}


def _run(clip_id: int, label: str, fn, *args) -> None:
    _jobs[clip_id] = label
    try:
        with _lock:                          # one heavy job at a time (GPU/CPU)
            fn(*args)
    finally:
        _jobs.pop(clip_id, None)


def _view(r) -> dict:
    d = dict(r)
    d.pop("embedding", None)
    for k in ("captions", "hashtags", "thumb_options", "segments"):
        d[k] = json.loads(d[k]) if d.get(k) else ([] if k != "captions" else {})
    d["job"] = _jobs.get(d["id"])
    d["caption_text"] = caption_text(d.get("hook") or "", d.get("description") or "", d["hashtags"])
    d["yt_description"] = ((d.get("description") or "") + "\n\n" + " ".join(d["hashtags"])).strip()
    return d


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    review = [_view(r) for r in pipe.db.clips(("rendered", "pending", "failed"))]
    done = [_view(r) for r in pipe.db.clips(("uploaded", "approved"))][:12]
    return templates.TemplateResponse(request, "index.html", {
        "review": review, "done": done, "status": pipe.status(), "jobs": _jobs,
        "grade": pipe.grade, "handoff_dir": str(cfg.path("paths", "handoff_dir")),
        "yt_mode": cfg["youtube"].get("mode", "manual")})


@app.get("/media/{clip_id}/{what}")
def media(clip_id: int, what: str):
    clip = pipe.db.clip(clip_id)
    if not clip:
        raise HTTPException(404)
    if what == "preview":
        p = clip["social_path"] or clip["master_path"]
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
    if not clip or clip["kind"] != "short":
        raise HTTPException(404)
    out_dir = Path(clip["master_path"] or cfg.sub("output")).parent / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = pipe.ensure_local(clip["source_id"])
    t = clip["start"] + clip["duration"] * 0.4
    if which == "before":
        out = out_dir / "before.jpg"
        if not out.exists():
            ff.extract_frame(src, t, out, vf="scale=1280:-2")
    else:
        out = out_dir / f"after_{int(strength * 100)}.jpg"
        if not out.exists():
            g = pipe.grade.with_strength(strength)
            g.grain = 0
            ff.extract_frame(src, t, out, vf=f"scale=1280:-2,{filter_chain(g, width=1280)}")
    return FileResponse(out)


@app.post("/clip/{clip_id}/save")
def save(clip_id: int, hook: str = Form(""), title: str = Form(""), description: str = Form(""),
         hashtags: str = Form("")):
    tags = [t if t.startswith("#") else f"#{t}" for t in hashtags.replace(",", " ").split() if t.strip()][:3]
    pipe.db.update_clip(clip_id, hook=hook.strip(), title=title.strip(), description=description.strip(),
                        hashtags=tags)
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/thumbnail")
def thumbnail(clip_id: int, word: str = Form(...), frame: int = Form(1)):
    pipe.rethumb(clip_id, word.strip().split()[0].upper() if word.strip() else "RIDE", frame)
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/rerender")
def rerender(clip_id: int, bg: BackgroundTasks, strength: float = Form(1.0), music: str = Form("")):
    pipe.db.update_clip(clip_id, grade_strength=max(0.0, min(1.0, strength)), music="1" if music else None)
    clip = pipe.db.clip(clip_id)
    fn = pipe.render_short_clip if clip["kind"] == "short" else pipe.render_long_clip
    bg.add_task(_run, clip_id, "re-rendering", fn, clip_id)
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/new-captions")
def new_captions(clip_id: int, bg: BackgroundTasks):
    clip = pipe.db.clip(clip_id)
    pipe.db.update_clip(clip_id, captions=None)
    fn = pipe.render_short_clip if clip["kind"] == "short" else pipe.render_long_clip
    bg.add_task(_run, clip_id, "writing new captions + re-rendering", fn, clip_id)
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/approve")
def approve(clip_id: int, bg: BackgroundTasks):
    def job():
        res = pipe.approve(clip_id)
        if "youtube_error" in res:
            pipe.db.update_clip(clip_id, error="YouTube upload failed: " + res["youtube_error"])
    bg.add_task(_run, clip_id, "uploading to YouTube (private)", job)
    return RedirectResponse(f"/#clip{clip_id}", status_code=303)


@app.post("/clip/{clip_id}/reject")
def reject(clip_id: int):
    pipe.db.update_clip(clip_id, status="rejected")
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
