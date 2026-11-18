"""Command line:  python -m moto_shorts <command>

  daily        ingest + analyze + render today's shorts (+ long video when ready)
  dashboard    open the review page (http://127.0.0.1:8765)
  ingest       only download/collect new footage
  analyze      only analyze new footage
  shorts N     pick and render N shorts now
  long         force-build a long video from unused footage
  check        verify FFmpeg, GPU encoder, keys and folders
  youtube-login  sign in to YouTube once (opens browser)
"""
from __future__ import annotations

import os
import sys

from .config import load_config


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "help"
    cfg = load_config()
    if cmd == "dashboard":
        import uvicorn
        import webbrowser
        d = cfg["dashboard"]
        webbrowser.open(f"http://{d['host']}:{d['port']}")
        uvicorn.run("moto_shorts.dashboard.app:app", host=d["host"], port=int(d["port"]))
        return 0
    if cmd == "check":
        return check(cfg)
    from .pipeline import Pipeline
    p = Pipeline(cfg)
    if cmd == "daily":
        p.daily()
    elif cmd == "startup":            # run at sign-in: only once per (local) day
        if p.ran_today():
            print("already ran today - skipping (use 'daily' or the dashboard's Run now to force)")
        else:
            p.daily()
    elif cmd == "ingest":
        print(f"added {p.ingest()} files")
    elif cmd == "analyze":
        p.analyze_pending()
    elif cmd == "shorts":
        for cid in p.pick_shorts(int(argv[1]) if len(argv) > 1 else 3):
            p.render_short_clip(cid)
    elif cmd == "long":
        print("long video:", p.maybe_build_long(force=True))
    elif cmd == "youtube-login":
        from .publish.youtube import _service
        _service(cfg.path("youtube", "client_secrets"), cfg.path("youtube", "token_file"))
        print("YouTube connected.")
    else:
        print(__doc__)
    return 0


def check(cfg) -> int:
    from . import ffmpeg as ff
    ok = True

    def line(good: bool, text: str) -> None:
        nonlocal ok
        ok &= good
        print(("  OK   " if good else "  ---  ") + text)

    print("Moto Shorts setup check")
    try:
        enc = ff.pick_encoder(cfg["render"].get("encoder", "auto"))
        line(True, f"FFmpeg found, video encoder: {enc}")
        line(enc in ("hevc_nvenc", "hevc_videotoolbox"), "GPU encoding (NVIDIA/Apple) available - much faster renders")
    except Exception as e:
        line(False, f"FFmpeg problem: {e}")
    line(bool(os.environ.get("ANTHROPIC_API_KEY")), "ANTHROPIC_API_KEY set (captions)")
    line(bool(os.environ.get("GOPRO_ACCESS_TOKEN")), "GOPRO_ACCESS_TOKEN set (GoPro Cloud)")
    if cfg["youtube"].get("mode", "manual") == "api":
        line(cfg.path("youtube", "client_secrets").exists(), "YouTube client secret file present")
        line(cfg.path("youtube", "token_file").exists(), "YouTube signed in (run: python -m moto_shorts youtube-login)")
    else:
        print("  OK   YouTube: manual upload mode (no Google setup needed)")
    from .thumbnails.make import find_font
    try:
        line(True, f"Thumbnail font: {find_font(cfg.path('thumbnails', 'font'))}")
    except FileNotFoundError as e:
        line(False, str(e))
    print("All good." if ok else "Some items need attention (see README).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
