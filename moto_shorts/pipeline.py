"""The daily pipeline: ingest -> analyze -> pick -> captions -> render -> thumbnails -> queue."""
from __future__ import annotations

import json
import random
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import ffmpeg as ff
from .analyze.audio import analyze_audio
from .analyze.proxy import make_proxies
from .analyze.score import Analysis, Candidate, combine, select, windows
from .analyze.telemetry import read_telemetry
from .analyze.video import analyze_video
from .captions.generate import CaptionSet, fallback_captions, generate_captions
from .config import Config
from .db import DB, now
from .edit.grade import Grade
from .edit.render import (RenderSettings, Segment, normalization_for, render_long, render_short,
                          render_social)
from .ingest.gopro_cloud import GoProCloud, TokenExpired, inbox_files, move_into
from .thumbnails.make import grab_graded_frames, make_thumbnail


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = DB(cfg.work_dir / "moto_shorts.db")
        self.grade = Grade.load(cfg.path("style", "preset"))
        self.status_file = cfg.work_dir / "status.json"

    # ------------------------------------------------------------------ status
    def set_status(self, **kv) -> None:
        data = self.status()
        data.update(kv)
        self.status_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def status(self) -> dict:
        try:
            return json.loads(self.status_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ------------------------------------------------------------------ ingest
    def ingest(self) -> int:
        raw = self.cfg.sub("raw")
        added = 0
        for f in inbox_files(self.cfg.path("paths", "inbox_dir")):
            key = f"local:{f.name}:{f.stat().st_size}"
            if self.db.has_source(key):
                continue
            dest = move_into(f, raw)
            info = ff.probe(dest)
            self.db.add_source(key, dest.name, str(dest), info.creation_time, info.duration)
            added += 1
            log(f"inbox: added {dest.name}")
        gcfg = self.cfg["gopro"]
        if gcfg.get("enabled"):
            try:
                cloud = GoProCloud()
                media = cloud.list_videos(int(gcfg.get("only_videos_newer_than_days", 30)))
                new = [m for m in media if not self.db.has_source(f"gopro:{m['id']}")]
                for m in new[: int(gcfg.get("max_downloads_per_run", 10))]:
                    log(f"GoPro Cloud: downloading {m.get('filename')} ...")
                    path = cloud.download(m["id"], raw)
                    info = ff.probe(path)
                    self.db.add_source(f"gopro:{m['id']}", path.name, str(path), m.get("captured_at"), info.duration)
                    added += 1
                self.set_status(gopro_ok=True, gopro_error=None)
            except TokenExpired as e:
                log(f"GoPro Cloud: {e}")
                self.set_status(gopro_ok=False, gopro_error=str(e))
            except Exception as e:                       # network etc. - inbox still works
                log(f"GoPro Cloud error: {e}")
                self.set_status(gopro_ok=False, gopro_error=f"{type(e).__name__}: {e}")
        return added

    def ensure_local(self, source_id: int) -> Path:
        """Raw file for a source; re-downloads from GoPro Cloud if it was cleaned up."""
        src = self.db.source(source_id)
        p = Path(src["path"])
        if p.exists():
            return p
        if src["gopro_id"].startswith("gopro:"):
            log(f"re-downloading {src['filename']} from GoPro Cloud")
            p = GoProCloud().download(src["gopro_id"].split(":", 1)[1], self.cfg.sub("raw"))
            self.db.update_source(source_id, path=str(p), status="analyzed")
            return p
        raise FileNotFoundError(f"Raw file gone and not in GoPro Cloud: {p}")

    def cleanup_raw(self) -> None:
        days = int(self.cfg["paths"].get("keep_raw_days", 21))
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for s in self.db.sources("analyzed"):
            added = datetime.fromisoformat(s["added_at"])
            p = Path(s["path"])
            if added < cutoff and p.exists() and s["gopro_id"].startswith("gopro:"):
                p.unlink()
                log(f"cleaned up raw file {p.name} (still in GoPro Cloud)")

    # ------------------------------------------------------------------ analyze
    def analyze_pending(self) -> None:
        for s in self.db.sources("new"):
            try:
                log(f"analyzing {s['filename']} ...")
                video_proxy, wav = make_proxies(s["path"], self.cfg.sub("proxies"))
                a = analyze_audio(wav)
                v = analyze_video(video_proxy)
                t = read_telemetry(s["path"])
                an = combine(a, v, t, s["duration"])
                out = self.cfg.sub("analysis") / f"source_{s['id']}.npz"
                np.savez_compressed(out, duration=an.duration, total=an.total, audio=an.audio, scene=an.scene,
                                    telem=an.telem, quality=an.quality, embeddings=an.embeddings,
                                    onset_times=an.onset_times,
                                    speed=t.speed_kmh, climb=t.climb)
                self.db.update_source(s["id"], status="analyzed", analysis_path=str(out))
                wav.unlink(missing_ok=True)
            except Exception as e:
                log(f"analysis failed for {s['filename']}: {e}")
                self.db.update_source(s["id"], status="failed", error=str(e))

    def load_analysis(self, source_id: int) -> tuple[Analysis, dict]:
        d = np.load(self.db.source(source_id)["analysis_path"])
        an = Analysis(float(d["duration"]), d["total"], d["audio"], d["scene"], d["telem"], d["quality"],
                      d["embeddings"], d["onset_times"])
        return an, {"speed": d["speed"], "climb": d["climb"]}

    # ------------------------------------------------------------------ picking
    def _free_candidates(self, min_len: int, max_len: int) -> tuple[list[Candidate], dict[int, np.ndarray]]:
        used = self.db.used_ranges()
        cands, onsets = [], {}
        for s in self.db.sources("analyzed"):
            an, _ = self.load_analysis(s["id"])
            onsets[s["id"]] = an.onset_times
            for c in windows(an, s["id"], min_len, max_len):
                if any(c.start < e + 2 and b < c.end + 2 for b, e in used.get(s["id"], [])):
                    continue
                cands.append(c)
        return cands, onsets

    def pick_shorts(self, k: int) -> list[int]:
        d = self.cfg["daily"]
        cands, onsets = self._free_candidates(int(d["short_min_seconds"]), int(d["short_max_seconds"]))
        picks = select(cands, k, self.db.recent_embeddings(), onsets=onsets)
        ids = []
        for c in picks:
            ids.append(self.db.add_clip("short", c.source_id, c.start, c.duration, c.score, c.embedding))
            log(f"picked short: source {c.source_id} @ {c.start:.0f}s for {c.duration:.0f}s (score {c.score:.2f})")
        if len(picks) < k:
            log(f"only {len(picks)} good unused moments left - add more footage")
        return ids

    # ------------------------------------------------------------------ captions
    def _stats_for(self, source_id: int, start: float, dur: float) -> dict:
        an, extra = self.load_analysis(source_id)
        s, e = int(start), int(start + dur)
        st = {"duration_s": round(dur), "exhaust_intensity_0to1": round(float(an.audio[s:e].mean()), 2),
              "scenery_0to1": round(float(an.scene[s:e].mean()), 2)}
        if len(extra["speed"]):
            st["avg_speed_kmh"] = round(float(np.mean(extra["speed"][s:e])) if e <= len(extra["speed"]) else 0)
            st["climbing"] = bool(np.mean(extra["climb"][s:e]) > 0.3) if e <= len(extra["climb"]) else False
        cap = self.db.source(source_id)["captured_at"]
        if cap:
            st["captured_at_local_hour"] = cap[11:13]
        return st

    def _recent_hooks(self) -> list[str]:
        return [r["hook"] for r in self.db.clips(("rendered", "approved", "uploaded", "posted")) if r["hook"]]

    def captions_for(self, frames: list[Path], stats: dict, kind: str) -> CaptionSet:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            log("no ANTHROPIC_API_KEY - using placeholder captions")
            return fallback_captions(self.cfg["captions"]["hashtag_pool"], kind)
        return generate_captions(self.cfg, frames, stats, kind, self._recent_hooks())

    # ------------------------------------------------------------------ render
    def _music(self, enabled: bool) -> Path | None:
        if not enabled:
            return None
        d = self.cfg.path("style", "music_dir")
        tracks = [p for p in d.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")] if d.exists() else []
        return random.choice(tracks) if tracks else None

    def render_short_clip(self, clip_id: int, regenerate_captions: bool = False) -> None:
        clip = self.db.clip(clip_id)
        out_dir = self.cfg.sub("output") / f"short_{clip_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            src = self.ensure_local(clip["source_id"])
            grade = self.grade.with_strength(clip["grade_strength"] if clip["grade_strength"] is not None else 1.0)
            start, dur = clip["start"], clip["duration"]

            frames = grab_graded_frames(src, [start + dur * f for f in (0.15, 0.4, 0.65, 0.9)],
                                        out_dir / "frames", grade)
            if not clip["captions"] or regenerate_captions:
                caps = self.captions_for(frames, self._stats_for(clip["source_id"], start, dur), "short")
                self.db.update_clip(clip_id, captions=caps.to_json(), hook=caps.hooks[0], title=caps.title,
                                    description=caps.description, hashtags=caps.hashtags,
                                    thumb_word=caps.thumbnail_words[0])
                clip = self.db.clip(clip_id)

            hook_file = None
            if self.cfg["captions"].get("show_hook_on_video") and clip["hook"]:
                hook_file = out_dir / "hook.txt"
                hook_file.write_text(clip["hook"], encoding="utf-8")
            from .thumbnails.make import find_font
            font = find_font(self.cfg.path("thumbnails", "font"))

            settings = RenderSettings.from_config(self.cfg)
            norm = normalization_for(src, start, dur, grade)
            log(f"rendering short {clip_id} ({settings.encoder}) ...")
            master = render_short(src, start, dur, out_dir / "youtube_short_4k.mp4", settings, grade, norm,
                                  hook_file, font, self._music(bool(clip["music"])))
            social = render_social(master, out_dir / "social_1080.mp4", settings)
            thumbs = self._thumbs(frames, clip["thumb_word"] or "RIDE", out_dir, vertical=False)
            self.db.update_clip(clip_id, status="rendered", master_path=str(master), social_path=str(social),
                                thumb_options=[str(f) for f in frames], thumb_path=str(thumbs), error=None)
        except Exception as e:
            log(f"short {clip_id} failed: {e}")
            self.db.update_clip(clip_id, status="failed", error=f"{e}\n{traceback.format_exc()[-1500:]}")

    def _thumbs(self, frames: list[Path], word: str, out_dir: Path, vertical: bool, frame_index: int = 1) -> Path:
        t = self.cfg["thumbnails"]
        size = (1080, 1920) if vertical else (1280, 720)
        return make_thumbnail(frames[min(frame_index, len(frames) - 1)], word, out_dir / "thumbnail.jpg",
                              self.cfg.path("thumbnails", "font"), float(t.get("letter_spacing", 0.35)),
                              float(t.get("word_height", 0.075)), size)

    def rethumb(self, clip_id: int, word: str, frame_index: int) -> None:
        clip = self.db.clip(clip_id)
        frames = [Path(p) for p in json.loads(clip["thumb_options"] or "[]")]
        if not frames:
            return
        out_dir = Path(clip["master_path"]).parent
        p = self._thumbs(frames, word, out_dir, vertical=False, frame_index=frame_index)
        self.db.update_clip(clip_id, thumb_word=word, thumb_path=str(p))

    # ------------------------------------------------------------------ long video
    def unused_good_seconds(self) -> float:
        lcfg = self.cfg["longform"]
        cands, _ = self._free_candidates(int(lcfg["segment_min_seconds"]), int(lcfg["segment_max_seconds"]))
        if not cands:
            return 0.0
        threshold = np.percentile([c.score for c in cands], 60)
        picks = select([c for c in cands if c.score >= threshold], 50, similarity_limit=0.995)
        return float(sum(c.duration for c in picks))

    def maybe_build_long(self, force: bool = False) -> int | None:
        lcfg = self.cfg["longform"]
        if not lcfg.get("enabled"):
            return None
        if self.db.clips(("pending", "rendered"), kind="long") and not force:
            return None                                   # one waiting for approval already
        available = self.unused_good_seconds()
        if available < float(lcfg["min_unused_minutes"]) * 60 and not force:
            log(f"long video: {available / 60:.1f} min of good unused footage (need {lcfg['min_unused_minutes']})")
            return None
        cands, onsets = self._free_candidates(int(lcfg["segment_min_seconds"]), int(lcfg["segment_max_seconds"]))
        target = float(lcfg["target_minutes"]) * 60
        picks: list[Candidate] = []
        for c in select(cands, 40, similarity_limit=0.97, onsets=onsets):
            if sum(p.duration for p in picks) >= target:
                break
            picks.append(c)
        if not picks:
            return None
        # chronological = feels like one ride
        picks.sort(key=lambda c: (self.db.source(c.source_id)["captured_at"] or "", c.start))
        segs = [{"source_id": c.source_id, "start": c.start, "duration": c.duration} for c in picks]
        clip_id = self.db.add_clip("long", None, 0, sum(c.duration for c in picks),
                                   float(np.mean([c.score for c in picks])), None, segs)
        self.render_long_clip(clip_id)
        return clip_id

    def render_long_clip(self, clip_id: int) -> None:
        clip = self.db.clip(clip_id)
        out_dir = self.cfg.sub("output") / f"long_{clip_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            grade = self.grade.with_strength(clip["grade_strength"] or 1.0)
            segs = []
            for i, s in enumerate(json.loads(clip["segments"])):
                src = self.ensure_local(s["source_id"])
                segs.append(Segment(src, s["start"], s["duration"], normalization_for(src, s["start"], s["duration"], grade),
                                    title=f"Section {i + 1}"))
            frame_srcs = sorted(segs, key=lambda s: -s.duration)[:4]
            frames = []
            for i, s in enumerate(frame_srcs):
                frames += grab_graded_frames(s.src, [s.start + s.duration * 0.5], out_dir / f"frames{i}", grade)
            if not clip["captions"]:
                caps = self.captions_for(frames, {"duration_min": round(clip["duration"] / 60, 1),
                                                  "segments": len(segs)}, "long")
                self.db.update_clip(clip_id, captions=caps.to_json(), hook=caps.hooks[0], title=caps.title,
                                    description=caps.description, hashtags=caps.hashtags,
                                    thumb_word=caps.thumbnail_words[0])
                clip = self.db.clip(clip_id)
            settings = RenderSettings.from_config(self.cfg)
            log(f"rendering long video {clip_id} ({len(segs)} segments, {clip['duration'] / 60:.1f} min) ...")
            master, chapters = render_long(segs, out_dir / "youtube_long_4k.mp4", settings, grade)
            thumb = self._thumbs(frames, clip["thumb_word"] or "RIDE", out_dir, vertical=False, frame_index=0)
            (out_dir / "chapters.json").write_text(json.dumps(chapters), encoding="utf-8")
            desc = clip["description"] or ""
            if len(chapters) >= 3 and "0:00" not in desc:   # YouTube chapters: first must be 0:00
                desc += "\n\n" + "\n".join(f"{int(t // 60)}:{int(t % 60):02d} {name}" for t, name in chapters)
                self.db.update_clip(clip_id, description=desc)
            self.db.update_clip(clip_id, status="rendered", master_path=str(master), thumb_path=str(thumb),
                                thumb_options=[str(f) for f in frames], error=None)
        except Exception as e:
            log(f"long video {clip_id} failed: {e}")
            self.db.update_clip(clip_id, status="failed", error=f"{e}\n{traceback.format_exc()[-1500:]}")

    # ------------------------------------------------------------------ approve
    def approve(self, clip_id: int) -> dict:
        """Upload to YouTube as PRIVATE and drop the phone package for TikTok/IG."""
        from .publish import youtube
        from .publish.handoff import handoff

        clip = self.db.clip(clip_id)
        tags = json.loads(clip["hashtags"] or "[]")
        result: dict = {}
        chapters = None              # long-video chapters are already in the description
        if self.cfg["youtube"].get("mode", "manual") == "api":
            try:
                vid = youtube.upload(self.cfg, Path(clip["master_path"]), clip["title"] or clip["hook"],
                                     clip["description"] or "", tags,
                                     Path(clip["thumb_path"]) if clip["thumb_path"] else None, chapters)
                result["youtube_id"] = vid
                self.db.update_clip(clip_id, youtube_id=vid)
            except Exception as e:
                result["youtube_error"] = str(e)
        if clip["kind"] == "short" and clip["social_path"]:
            folder = handoff(self.cfg.path("paths", "handoff_dir"), clip_id, Path(clip["social_path"]),
                             Path(clip["thumb_path"]) if clip["thumb_path"] else None,
                             clip["hook"] or "", clip["description"] or "", tags)
            result["handoff"] = str(folder)
        status = "uploaded" if "youtube_id" in result else "approved"
        self.db.update_clip(clip_id, status=status, posted_at=now())
        return result

    # ------------------------------------------------------------------ daily
    def daily(self) -> None:
        self.set_status(last_run_started=now(), running=True)
        try:
            self.ingest()
            self.analyze_pending()
            waiting = len(self.db.clips(("pending", "rendered"), kind="short"))
            k = max(int(self.cfg["daily"]["shorts_per_day"]) - max(waiting - 6, 0), 0)
            for cid in self.pick_shorts(k):
                self.render_short_clip(cid)
            self.maybe_build_long()
            self.cleanup_raw()
        finally:
            self.set_status(last_run_finished=now(), running=False)
        log("daily run finished - open the dashboard to review")
