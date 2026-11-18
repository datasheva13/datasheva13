"""Measure the look of reference videos and write a matching grade preset.

Usage (on your computer, not in the cloud):
    python tools/analyze_refs.py                                   # your 3 videos (75%) + ScramblerPOV (25%)
    python tools/analyze_refs.py URL@START ...                     # specific videos (START in seconds)
    python tools/analyze_refs.py https://www.youtube.com/@ScramblerPOV/videos   # a whole channel:
                                                                   # samples its 8 latest videos

Downloads ~40 s of each video (from the timestamp you liked) with yt-dlp, samples
frames, and measures: black level, highlight level, contrast, saturation, shadow and
highlight tint, average shot length. Writes presets/reference_look.yaml, a blend of
those numbers translated into our grade parameters, and prints a short report.
The references are only used for measurement; nothing is re-used in your videos.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from moto_shorts.edit.grade import Grade  # noqa: E402

CHANNEL_VIDEOS = 8

# Your 3 videos are the main look; ScramblerPOV is a light influence (not copied).
MAIN_REFERENCES = [
    "https://www.youtube.com/watch?v=c7Dpmg-PI4c@132",
    "https://www.youtube.com/watch?v=tFxxQcsjBO0@295",
    "https://www.youtube.com/watch?v=0fMoZ0-cAY4@33",
]
INFLUENCE_REFERENCES = [
    "https://www.youtube.com/@ScramblerPOV/videos",
]
MAIN_WEIGHT = 0.75          # share of the final look from MAIN_REFERENCES
INFLUENCE_WEIGHT = 0.25     # share from INFLUENCE_REFERENCES (all sampled videos together)
SECONDS = 40


def download(url: str, start: int, out_dir: Path) -> Path:
    out = out_dir / f"ref_{abs(hash(url)) % 10**8}.mp4"
    if out.exists():
        return out
    subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
                    "--download-sections", f"*{start}-{start + SECONDS}", "--force-keyframes-at-cuts",
                    "-o", str(out), url], check=True)
    return out


def frames(path: Path, fps: float = 2, w: int = 480) -> np.ndarray:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:-2",
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip().split(",")
    h = int(round(int(probe[1]) * w / int(probe[0]) / 2) * 2)
    n = len(proc.stdout) // (w * h * 3)
    return np.frombuffer(proc.stdout[: n * w * h * 3], np.uint8).reshape(n, h, w, 3).astype(np.float32) / 255


def shot_lengths(path: Path) -> float:
    f = frames(path, fps=6, w=160)
    if len(f) < 3:
        return 0.0
    diffs = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3))
    cuts = np.sum(diffs > max(0.12, np.percentile(diffs, 97)))
    return len(f) / 6 / max(cuts + 1, 1)


def measure(f: np.ndarray) -> dict:
    px = f.reshape(-1, 3)
    # ignore letterbox bars
    luma = px @ np.array([0.2126, 0.7152, 0.0722])
    px, luma = px[luma > 0.01], luma[luma > 0.01]
    mx, mn = px.max(axis=1), px.min(axis=1)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    sh, hi = luma < 0.25, luma > 0.7

    def tint(mask):
        if mask.sum() < 100:
            return [0.0, 0.0, 0.0]
        m = px[mask].mean(axis=0)
        return list(m - m.mean())

    return {"black": float(np.percentile(luma, 1.5)), "white": float(np.percentile(luma, 99)),
            "mid": float(np.median(luma)), "contrast": float(luma.std()), "saturation": float(sat.mean()),
            "shadow_tint": tint(sh), "highlight_tint": tint(hi)}


def to_grade(m: dict, base: Grade) -> Grade:
    g = Grade(**base.__dict__)
    g.name = "reference_look"
    black = float(np.clip(m["black"], 0.0, 0.08))
    white = float(np.clip(m["white"], 0.88, 1.0))
    g.curve_master = [[0.0, round(black, 3)], [0.25, round(0.2 + black * 0.3, 3)], [0.5, 0.5],
                      [0.75, round(0.75 + (white - 0.9) * 0.4 + 0.03, 3)], [1.0, round(white, 3)]]
    # Typical well-exposed Rec.709 ride footage: contrast ~0.21, saturation ~0.33
    g.base_contrast = round(float(np.clip(1.0 + (m["contrast"] - 0.21) * 1.5 + 0.08, 1.0, 1.25)), 3)
    g.saturation = round(float(np.clip(m["saturation"] / 0.33, 0.8, 1.08)), 3)
    k = 1.6       # pixel tint -> colorbalance amount
    g.shadows_rgb = [round(float(np.clip(v * k, -0.08, 0.08)), 3) for v in m["shadow_tint"]]
    g.highlights_rgb = [round(float(np.clip(v * k, -0.08, 0.08)), 3) for v in m["highlight_tint"]]
    return g


def expand(refs: list[str]) -> list[str]:
    """Channel links (/@name or /@name/videos) -> their latest videos, sampled from 1:00 in."""
    out = []
    for ref in refs:
        if "/@" in ref or "/channel/" in ref:
            url = ref if ref.rstrip("/").endswith("/videos") else ref.rstrip("/") + "/videos"
            ids = subprocess.run([sys.executable, "-m", "yt_dlp", "--flat-playlist", "--print", "id",
                                  "--playlist-end", str(CHANNEL_VIDEOS), url],
                                 capture_output=True, text=True).stdout.split()
            out += [f"https://www.youtube.com/watch?v={i}@60" for i in ids]
            print(f"{ref}: sampling {len(ids)} videos")
        else:
            out.append(ref)
    return out


def _mean(stats: list[dict]) -> dict:
    return {k: (float(np.mean([s[k] for s in stats])) if not isinstance(stats[0][k], list)
                else list(np.mean([s[k] for s in stats], axis=0))) for k in stats[0]}


def blend(groups: list[tuple[list[dict], float]]) -> dict:
    """Weighted average of group averages: a group of 8 videos counts no more than its weight."""
    groups = [(st, w) for st, w in groups if st and w > 0]
    total = sum(w for _, w in groups)
    means = [(_mean(st), w / total) for st, w in groups]
    out = {}
    for k in means[0][0]:
        if isinstance(means[0][0][k], list):
            out[k] = list(np.sum([np.array(m[k]) * w for m, w in means], axis=0))
        else:
            out[k] = float(sum(m[k] * w for m, w in means))
    return out


def measure_group(refs: list[str], work: Path, shots: list[float]) -> list[dict]:
    stats = []
    for ref in expand(refs):
        url, _, start = ref.partition("@")
        print(f"Downloading {SECONDS}s of {url} from {start or 0}s ...")
        try:
            p = download(url, int(start or 0), work)
        except subprocess.CalledProcessError:
            print("  (could not download, skipped)")
            continue
        m = measure(frames(p))
        s = shot_lengths(p)
        stats.append(m)
        shots.append(s)
        print(f"  black {m['black']:.3f}  white {m['white']:.3f}  contrast {m['contrast']:.3f}  "
              f"saturation {m['saturation']:.3f}  avg shot {s:.1f}s")
    return stats


def main(args: list[str]) -> None:
    work = Path(tempfile.gettempdir()) / "moto_refs"
    work.mkdir(exist_ok=True)
    shots: list[float] = []
    if args:                                  # links given on the command line: equal weight
        groups = [(measure_group(args, work, shots), 1.0)]
    else:
        print(f"Main look ({int(MAIN_WEIGHT * 100)}%):")
        main_stats = measure_group(MAIN_REFERENCES, work, shots)
        print(f"Influence ({int(INFLUENCE_WEIGHT * 100)}%):")
        groups = [(main_stats, MAIN_WEIGHT), (measure_group(INFLUENCE_REFERENCES, work, shots), INFLUENCE_WEIGHT)]
    if not any(st for st, _ in groups):
        print("Nothing could be downloaded - check your internet connection / the links.")
        return
    avg = blend(groups)
    base = Grade.load(ROOT / "presets" / "channel_look.yaml")
    g = to_grade(avg, base)
    out = ROOT / "presets" / "reference_look.yaml"
    g.save(out)
    print(f"\nWrote {out}")
    print(f"Average shot length in references: {np.mean(shots):.1f}s "
          f"(use it for longform.segment_min_seconds if you want similar pacing)")
    print("To use it: set  style.preset: presets/reference_look.yaml  in config.yaml,")
    print("then compare in the dashboard with the before/after slider.")


if __name__ == "__main__":
    main(sys.argv[1:])
