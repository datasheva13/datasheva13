"""The daily pipeline: ingest -> analyze -> bike check -> pick -> captions -> render -> queue -> cleanup."""
from __future__ import annotations

import json
import os
import random
import shutil
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import ffmpeg as ff
from .analyze.audio import analyze_audio
from .analyze.bikecheck import check_bike, per_second_factors
from .analyze.proxy import make_proxies
from .analyze.score import Analysis, Candidate, combine, select, windows
from .analyze.telemetry import read_telemetry
from .analyze.video import analyze_video
from .captions.generate import CaptionSet, fallback_captions, generate_captions
from .config import Config
from .db import DB, now
from .edit.grade import Grade
from .edit.render import (RenderSettings, Segment, normalization_for, render_instagram_rotated, render_long,
                          render_short, render_social)
from .ingest.gopro_cloud import GoProCloud, TokenExpired, inbox_files, move_into
from .thumbnails.make import find_font, grab_graded_frames, make_thumbnail

POSTED_KEEP_DAYS = 7


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _age_days(iso: str | None) -> float:
    if not iso:
        return 0.0
    t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 86400


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
                media = cloud.list_videos(int(gcfg.get("only_videos_newer_than_days", 0)),
                                          max_pages=int(gcfg.get("max_pages", 40)))
                new = [m for m in media if not self.db.has_source(f"gopro:{m['id']}")]
                for m in self._download_order(new, int(gcfg.get("max_downloads_per_run", 3))):
                    log(f"GoPro Cloud: downloading {m.get('filename')} ({(m.get('captured_at') or '')[:10]}) ...")
                    path = cloud.download(m["id"], raw)
                    info = ff.probe(path)
                    self.db.add_source(f"gopro:{m['id']}", path.name, str(path), m.get("captured_at"), info.duration)
                    added += 1
                self.set_status(gopro_ok=True, gopro_error=None, gopro_library=len(media), gopro_not_downloaded=len(new))
            except TokenExpired as e:
                log(f"GoPro Cloud: {e}")
                self.set_status(gopro_ok=False, gopro_error=str(e))
            except Exception as e:                       # network etc. - inbox still works
                log(f"GoPro Cloud error: {e}")
                self.set_status(gopro_ok=False, gopro_error=f"{type(e).__name__}: {e}")
        return added

    @staticmethod
    def _download_order(new: list[dict], k: int) -> list[dict]:
        """Alternate between the newest rides and random older ones, so the whole
        library gets used (useful in winter when there's little new footage)."""
        if k <= 0 or not new:
            return []
        newest = sorted(new, key=lambda m: m.get("captured_at") or "", reverse=True)
        picks = newest[: (k + 1) // 2]
        older = [m for m in newest if m not in picks]
        picks += random.sample(older, min(k - len(picks), len(older)))
        return picks

    def ensure_local(self, source_id: int) -> Path:
        """Raw file for a source; re-downloads from GoPro Cloud if it was cleaned up."""
        src = self.db.source(source_id)
        p = Path(src["path"])
        if p.exists():
            return p
        if src["gopro_id"].startswith("gopro:"):
            log(f"re-downloading {src['filename']} from GoPro Cloud")
            p = GoProCloud().download(src["gopro_id"].split(":", 1)[1], self.cfg.sub("raw"))
            self.db.update_source(source_id, path=str(p), added_at=now())
            return p
        raise FileNotFoundError(f"Raw file gone and not in GoPro Cloud: {p}")

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
                self.bike_check(s["id"], video_proxy)
            except Exception as e:
                log(f"analysis failed for {s['filename']}: {e}")
                self.db.update_source(s["id"], status="failed", error=str(e))
        # retry bike checks that failed earlier (e.g. no internet)
        for s in self.db.sources("analyzed"):
            if s["origin"] == "auto" and not s["bike_check"]:
                proxy = self.cfg.sub("proxies") / f"{Path(s['path']).stem}_proxy.mp4"
                if proxy.exists():
                    self.bike_check(s["id"], proxy)

    def bike_check(self, source_id: int, proxy: Path) -> None:
        """Skip mountain-bike/non-motorcycle videos; mark moments without the bike in view."""
        s = self.db.source(source_id)
        if s["origin"] != "auto" or not _has_api_key():
            return
        n = int(np.clip((s["duration"] or 0) / 20, 8, 30))
        res = check_bike(proxy, s["duration"], self.cfg.sub("proxies") / f"check_{source_id}", n,
                         self.cfg["captions"].get("model", "claude-opus-5-5"))
        if "error" in res:
            log(f"bike check for {s['filename']} failed ({res['error']}) - will retry next run")
            return
        usable = 1 if res["is_motorcycle_ride"] else 0
        self.db.update_source(source_id, bike_check=json.dumps(res), usable=usable)
        if not usable:
            log(f"skipping {s['filename']}: not a motorcycle ride ({res.get('activity')})")
            p = Path(s["path"])
            if s["gopro_id"].startswith("gopro:") and p.exists():
                p.unlink()                       # no need to keep it on disk

    def load_analysis(self, source_id: int) -> tuple[Analysis, dict]:
        d = np.load(self.db.source(source_id)["analysis_path"])
        an = Analysis(float(d["duration"]), d["total"], d["audio"], d["scene"], d["telem"], d["quality"],
                      d["embeddings"], d["onset_times"])
        return an, {"speed": d["speed"], "climb": d["climb"]}

    # ------------------------------------------------------------------ picking
    def _free_candidates(self, min_len: int, max_len: int) -> tuple[list[Candidate], dict[int, np.ndarray]]:
        used = self.db.used_ranges()
        uses = self.db.source_use_counts()
        cands, onsets = [], {}
        for s in self.db.sources("analyzed"):
            if s["usable"] == 0 or s["origin"] == "upload":
                continue
            an, _ = self.load_analysis(s["id"])
            check = json.loads(s["bike_check"]) if s["bike_check"] else None
            vis, colors = per_second_factors(check, len(an.total))
            # bike must be in view; beautiful color matters; then the usual sound+scenery score
            an.total = (0.82 * an.total + 0.18 * colors) * vis
            onsets[s["id"]] = an.onset_times
            for c in windows(an, s["id"], min_len, max_len):
                if any(c.start < e + 2 and b < c.end + 2 for b, e in used.get(s["id"], [])):
                    continue
                if float(vis[int(c.start):int(c.end)].mean()) < 0.6:      # mostly no bike visible
                    continue
                c.score -= 0.03 * uses.get(s["id"], 0)                  # alternate between rides
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
        src = self.db.source(source_id)
        st: dict = {"duration_s": round(dur)}
        if src["analysis_path"]:
            an, extra = self.load_analysis(source_id)
            s, e = int(start), int(start + dur)
            st.update({"exhaust_intensity_0to1": round(float(an.audio[s:e].mean()), 2),
                       "scenery_0to1": round(float(an.scene[s:e].mean()), 2)})
            if len(extra["speed"]) >= e > s:
                st["avg_speed_kmh"] = round(float(np.mean(extra["speed"][s:e])))
                st["climbing"] = bool(np.mean(extra["climb"][s:e]) > 0.3)
        if src["captured_at"]:
            st["captured"] = src["captured_at"][:16]
        return st

    def _history(self) -> dict:
        return {"hooks": self.db.history("hook"), "titles": self.db.history("title"),
                "words_used": self.db.posted_words() + self.db.history("thumb_word", 30)}

    def captions_for(self, frames: list[Path], stats: dict, kind: str) -> CaptionSet:
        if not _has_api_key():
            log("no ANTHROPIC_API_KEY - using placeholder captions")
            h = self._history()
            return fallback_captions(self.cfg["captions"]["hashtag_pool"], kind, "no API key",
                                     h["words_used"], h["hooks"])
        caps = generate_captions(self.cfg, frames, stats, kind, self._history())
        if caps.note:
            log(caps.note)
        return caps

    def _save_captions(self, clip_id: int, caps: CaptionSet) -> None:
        self.db.update_clip(clip_id, captions=caps.to_json(), hook=caps.hooks[0], title=caps.title,
                            description=caps.description, hashtags=caps.hashtags,
                            thumb_word=caps.thumbnail_words[0], captions_note=caps.note or None)

    # ------------------------------------------------------------------ render
    def _music(self, enabled: bool) -> Path | None:
        if not enabled:
            return None
        d = self.cfg.path("style", "music_dir")
        tracks = [p for p in d.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")] if d.exists() else []
        return random.choice(tracks) if tracks else None

    def _loop_split(self, source_id: int, start: float, dur: float) -> float:
        """Where the loop is cut: on a throttle hit 40-65% into the clip, so the short opens
        on sound and the hidden cut lands on a sharp moment."""
        lo, hi = start + dur * 0.40, start + dur * 0.65
        src = self.db.source(source_id)
        if src["analysis_path"]:
            an, _ = self.load_analysis(source_id)
            hits = an.onset_times[(an.onset_times >= lo) & (an.onset_times <= hi)]
            if len(hits):
                return float(hits[0]) - 0.05
        return start + dur * 0.5

    def render_short_clip(self, clip_id: int) -> None:
        clip = self.db.clip(clip_id)
        out_dir = self.cfg.sub("output") / f"short_{clip_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            src = self.ensure_local(clip["source_id"])
            grade = self.grade.with_strength(clip["grade_strength"] if clip["grade_strength"] is not None else 1.0)
            start, dur = clip["start"], clip["duration"]

            frames = grab_graded_frames(src, [start + dur * f for f in (0.15, 0.4, 0.65, 0.9)],
                                        out_dir / "frames", grade)
            if not clip["captions"]:
                self._save_captions(clip_id, self.captions_for(frames, self._stats_for(clip["source_id"], start, dur),
                                                               "short"))
                clip = self.db.clip(clip_id)

            hook_file = None
            if self.cfg["captions"].get("show_hook_on_video") and clip["hook"]:
                hook_file = out_dir / "hook.txt"
                hook_file.write_text(clip["hook"], encoding="utf-8")
            font = find_font(self.cfg.path("thumbnails", "font"))

            settings = RenderSettings.from_config(self.cfg)
            norm = normalization_for(src, start, dur, grade)
            split = self._loop_split(clip["source_id"], start, dur) if clip["loop"] else None
            log(f"rendering short {clip_id} ({settings.encoder}) ...")
            music = self._music(bool(clip["music"]))
            master = render_short(src, start, dur, out_dir / "youtube_short_4k.mp4", settings, grade, norm,
                                  hook_file, font, music, loop_split=split)
            social = render_social(master, out_dir / "social_1080.mp4", settings)
            ig = social                                      # Instagram: same vertical layout as TikTok
            icfg = self.cfg.get("instagram", {}) or {}
            if icfg.get("layout", "vertical") == "rotated":
                ig = render_instagram_rotated(src, start, dur, out_dir / "instagram_rotated_1080.mp4", settings,
                                              grade, norm, hook_file, font, icfg.get("rotate", "clockwise"), music)
            thumbs = self._thumbs(frames, clip["thumb_word"] or "RIDE", out_dir, frame_index=1)
            self.db.update_clip(clip_id, status="rendered", master_path=str(master), social_path=str(social),
                                instagram_path=str(ig), thumb_options=[str(f) for f in frames],
                                thumb_path=str(thumbs), error=None)
        except Exception as e:
            log(f"short {clip_id} failed: {e}")
            self.db.update_clip(clip_id, status="failed", error=f"{e}\n{traceback.format_exc()[-1500:]}")

    def _thumbs(self, frames: list[Path], word: str, out_dir: Path, frame_index: int = 1) -> Path:
        t = self.cfg["thumbnails"]
        return make_thumbnail(frames[min(frame_index, len(frames) - 1)], word, out_dir / "thumbnail.jpg",
                              self.cfg.path("thumbnails", "font"), float(t.get("letter_spacing", 0.35)),
                              float(t.get("word_height", 0.075)), (1280, 720))

    def rethumb(self, clip_id: int, word: str, frame_index: int) -> None:
        clip = self.db.clip(clip_id)
        frames = [Path(p) for p in json.loads(clip["thumb_options"] or "[]")]
        if not frames:
            return
        p = self._thumbs(frames, word, Path(clip["master_path"]).parent, frame_index=frame_index)
        self.db.update_clip(clip_id, thumb_word=word, thumb_path=str(p))

    def trim(self, clip_id: int, start_delta: float, end_delta: float) -> None:
        """Move the start/end (seconds, + = later). Stays inside the source file, min 5 s."""
        clip = self.db.clip(clip_id)
        src = self.db.source(clip["source_id"])
        start = max(0.0, clip["start"] + start_delta)
        end = min(float(src["duration"] or 1e9), clip["start"] + clip["duration"] + end_delta)
        if end - start < 5:
            end = start + 5
        self.db.update_clip(clip_id, start=round(start, 2), duration=round(end - start, 2))

    # ------------------------------------------------------------------ your own uploads
    def add_upload(self, path: Path, kind: str) -> int:
        """kind 'short': blur layout + grade + all 3 platform exports (no loop re-cut, you edited it).
        kind 'long': colour grade only, 4K 16:9 for YouTube."""
        info = ff.probe(path)
        sid = self.db.add_source(f"upload:{path.name}:{path.stat().st_size}", path.name, str(path),
                                 info.creation_time, info.duration, origin="upload")
        self.db.update_source(sid, status="analyzed")
        if kind == "short":
            cid = self.db.add_clip("short", sid, 0.0, info.duration, 0.0, None)
            self.db.update_clip(cid, origin="upload", loop=0)
            self.render_short_clip(cid)
        else:
            cid = self.db.add_clip("long", None, 0.0, info.duration, 0.0, None,
                                   [{"source_id": sid, "start": 0.0, "duration": info.duration}])
            self.db.update_clip(cid, origin="upload")
            self.render_long_clip(cid)
        return cid

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
        picks.sort(key=lambda c: (self.db.source(c.source_id)["captured_at"] or "", c.start))   # feels like one ride
        segs = [{"source_id": c.source_id, "start": c.start, "duration": c.duration} for c in picks]
        clip_id = self.db.add_clip("long", None, 0, sum(c.duration for c in picks),
                                   float(np.mean([c.score for c in picks])), None, segs)
        self.render_long_clip(clip_id)
        return clip_id

    def render_long_clip(self, clip_id: int) -> None:
        clip = self.db.clip(clip_id)
        out_dir = self.cfg.sub("output") / f"long_{clip_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        upload = clip["origin"] == "upload"
        try:
            grade = self.grade.with_strength(clip["grade_strength"] if clip["grade_strength"] is not None else 1.0)
            segs = []
            for i, s in enumerate(json.loads(clip["segments"])):
                src = self.ensure_local(s["source_id"])
                segs.append(Segment(src, s["start"], s["duration"], normalization_for(src, s["start"], s["duration"], grade),
                                    title=f"Section {i + 1}"))
            if upload:
                s0 = segs[0]
                frames = grab_graded_frames(s0.src, [s0.start + s0.duration * f for f in (0.2, 0.4, 0.6, 0.8)],
                                            out_dir / "frames", grade)
            else:
                frames = []
                for i, s in enumerate(sorted(segs, key=lambda s: -s.duration)[:4]):
                    frames += grab_graded_frames(s.src, [s.start + s.duration * 0.5], out_dir / f"frames{i}", grade)
            if not clip["captions"]:
                self._save_captions(clip_id, self.captions_for(frames, {"duration_min": round(clip["duration"] / 60, 1),
                                                                        "segments": len(segs)}, "long"))
                clip = self.db.clip(clip_id)
            settings = RenderSettings.from_config(self.cfg)
            log(f"rendering long video {clip_id} ({len(segs)} segments, {clip['duration'] / 60:.1f} min) ...")
            master, chapters = render_long(segs, out_dir / "youtube_long_4k.mp4", settings, grade, fades=not upload)
            thumb = self._thumbs(frames, clip["thumb_word"] or "RIDE", out_dir, frame_index=0)
            (out_dir / "chapters.json").write_text(json.dumps(chapters), encoding="utf-8")
            desc = clip["description"] or ""
            if not upload and len(chapters) >= 3 and "0:00" not in desc:   # YouTube chapters: first must be 0:00
                desc += "\n\n" + "\n".join(f"{int(t // 60)}:{int(t % 60):02d} {name}" for t, name in chapters)
                self.db.update_clip(clip_id, description=desc)
            self.db.update_clip(clip_id, status="rendered", master_path=str(master), thumb_path=str(thumb),
                                thumb_options=[str(f) for f in frames], error=None)
        except Exception as e:
            log(f"long video {clip_id} failed: {e}")
            self.db.update_clip(clip_id, status="failed", error=f"{e}\n{traceback.format_exc()[-1500:]}")

    # ------------------------------------------------------------------ approve / reject
    def approve(self, clip_id: int) -> dict:
        """Mark as posted (kept 7 days in POSTED), prepare the phone package, optional YouTube API upload."""
        from .publish import youtube
        from .publish.handoff import handoff

        clip = self.db.clip(clip_id)
        tags = json.loads(clip["hashtags"] or "[]")
        result: dict = {}
        if self.cfg["youtube"].get("mode", "manual") == "api":
            try:
                vid = youtube.upload(self.cfg, Path(clip["master_path"]), clip["title"] or clip["hook"],
                                     clip["description"] or "", tags,
                                     Path(clip["thumb_path"]) if clip["thumb_path"] else None, None)
                result["youtube_id"] = vid
                self.db.update_clip(clip_id, youtube_id=vid)
            except Exception as e:
                result["youtube_error"] = str(e)
        if clip["kind"] == "short" and clip["social_path"]:
            folder = handoff(self.cfg.path("paths", "handoff_dir"), clip_id, Path(clip["social_path"]),
                             Path(clip["thumb_path"]) if clip["thumb_path"] else None,
                             clip["hook"] or "", clip["description"] or "", tags,
                             Path(clip["instagram_path"]) if clip["instagram_path"] else None)
            result["handoff"] = str(folder)
            self.db.update_clip(clip_id, handoff_path=str(folder))
        self.db.update_clip(clip_id, status="posted", posted_at=now())
        return result

    def _delete_files(self, clip) -> None:
        for p in (clip["master_path"], clip["thumb_path"]):
            if p:
                shutil.rmtree(Path(p).parent, ignore_errors=True)
                break
        out_dir = self.cfg.work_dir / "output" / f"{clip['kind']}_{clip['id']}"
        shutil.rmtree(out_dir, ignore_errors=True)
        if clip["handoff_path"]:
            shutil.rmtree(clip["handoff_path"], ignore_errors=True)
        if clip["origin"] == "upload":                     # your own upload: remove the copied file too
            ids = [clip["source_id"]] if clip["source_id"] else [s["source_id"] for s in json.loads(clip["segments"] or "[]")]
            for sid in ids:
                src = self.db.source(sid)
                if src:
                    Path(src["path"]).unlink(missing_ok=True)

    def reject(self, clip_id: int) -> None:
        """Gone immediately (files deleted); the moment is never suggested again."""
        clip = self.db.clip(clip_id)
        self._delete_files(clip)
        self.db.update_clip(clip_id, status="rejected", master_path=None, social_path=None, instagram_path=None,
                            thumb_path=None, thumb_options=None)

    # ------------------------------------------------------------------ cleanup
    def cleanup(self) -> None:
        days = int(self.cfg["paths"].get("cleanup_days", 30))
        # 1. posted for more than 7 days -> removed from the app and the computer
        for c in self.db.clips(("posted",)):
            if _age_days(c["posted_at"]) > POSTED_KEEP_DAYS:
                self._delete_files(c)
                self.db.update_clip(c["id"], status="archived", master_path=None, social_path=None,
                                    instagram_path=None, thumb_path=None, thumb_options=None)
                log(f"removed posted {c['kind']} {c['id']} (older than {POSTED_KEEP_DAYS} days)")
        # 2. downloaded GoPro footage older than N days (still in GoPro Cloud; re-downloaded if needed)
        pending_sources = {c["source_id"] for c in self.db.clips(("pending", "rendered", "failed")) if c["source_id"]}
        for s in self.db.sources():
            p = Path(s["path"])
            if (s["gopro_id"].startswith("gopro:") and p.exists() and _age_days(s["added_at"]) > days
                    and s["id"] not in pending_sources):
                p.unlink()
                log(f"removed raw file {p.name} (older than {days} days, still in GoPro Cloud)")
        # 3. leftover working files older than N days (proxies, failed renders)
        for folder in (self.cfg.sub("proxies"), self.cfg.sub("output")):
            for p in folder.iterdir():
                if _age_days(datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()) > days:
                    in_review = any(p.name == f"{c['kind']}_{c['id']}" for c in self.db.clips(("pending", "rendered")))
                    if not in_review:
                        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        # 4. old logs
        logs = self.cfg.work_dir / "logs"
        if logs.exists():
            for p in logs.glob("*.log"):
                if _age_days(datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()) > days:
                    p.unlink(missing_ok=True)

    # ------------------------------------------------------------------ daily
    def ran_today(self) -> bool:
        """True if a run already finished today in your time zone."""
        last = self.status().get("last_run_finished")
        if not last:
            return False
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(self.cfg["paths"].get("timezone", "America/Denver"))
        except Exception:
            tz = timezone.utc
        t = datetime.fromisoformat(last)
        return t.astimezone(tz).date() == datetime.now(tz).date()

    def daily(self) -> None:
        self.set_status(last_run_started=now(), running=True)
        try:
            self.cleanup()
            self.ingest()
            self.analyze_pending()
            waiting = len(self.db.clips(("pending", "rendered"), kind="short"))
            per_day = int(self.cfg["daily"]["shorts_per_day"])
            k = max(per_day - max(waiting - 3 * per_day, 0), 0)      # don't pile up if you're away
            for cid in self.pick_shorts(k):
                self.render_short_clip(cid)
            self.maybe_build_long()
        finally:
            self.set_status(last_run_finished=now(), running=False)
        log("daily run finished - open the dashboard to review")
