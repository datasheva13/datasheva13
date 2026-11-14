"""Thin FFmpeg/ffprobe helpers: probing, encoder selection, running commands."""
from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class FFmpegError(RuntimeError):
    pass


def run(args: list[str], quiet: bool = True) -> None:
    cmd = [FFMPEG, "-hide_banner", "-y"] + (["-loglevel", "error"] if quiet else []) + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}):\n{' '.join(cmd)}\n{proc.stderr[-4000:]}")


@dataclass
class MediaInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    pix_fmt: str
    data_streams: list[int]          # stream indexes of data tracks (GoPro GPMF lives here)
    creation_time: str | None

    @property
    def landscape(self) -> bool:
        return self.width >= self.height


def probe(path: str | Path) -> MediaInfo:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {out.stderr}")
    data = json.loads(out.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    num, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    data_streams = [
        s["index"] for s in data["streams"]
        if s["codec_type"] == "data" and "gpmd" in (s.get("codec_tag_string", "") + s.get("tags", {}).get("handler_name", "")).lower()
        or (s["codec_type"] == "data" and "GoPro MET" in s.get("tags", {}).get("handler_name", ""))
    ]
    fmt = data.get("format", {})
    return MediaInfo(
        path=Path(path),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=round(fps, 3),
        duration=float(fmt.get("duration") or video.get("duration") or 0),
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
        pix_fmt=video.get("pix_fmt", ""),
        data_streams=data_streams,
        creation_time=fmt.get("tags", {}).get("creation_time"),
    )


@functools.lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return {line.split()[1] for line in out.splitlines() if line.startswith(" V") and len(line.split()) > 1}


@functools.lru_cache(maxsize=4)
def _encoder_works(name: str) -> bool:
    """An encoder can be listed but unusable (no GPU / driver). Try a tiny encode."""
    try:
        run(["-f", "lavfi", "-i", "color=black:s=256x256:d=0.2", "-c:v", name, "-pix_fmt",
             "p010le" if "nvenc" in name else "yuv420p10le", "-f", "null", "-"])
        return True
    except FFmpegError:
        return False


def pick_encoder(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    for name in ("hevc_nvenc", "hevc_videotoolbox", "libx265"):
        if name in available_encoders() and _encoder_works(name):
            return name
    return "libx264"


def video_codec_args(encoder: str, bitrate: str) -> list[str]:
    """High-bitrate 10-bit HEVC (H.264 8-bit as a last resort)."""
    rate = bitrate.rstrip("Mm")
    peak = f"{int(float(rate) * 1.5)}M"
    buf = f"{int(float(rate) * 2)}M"
    if encoder == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-b:v", bitrate,
                "-maxrate", peak, "-bufsize", buf, "-profile:v", "main10", "-pix_fmt", "p010le",
                "-spatial-aq", "1", "-tag:v", "hvc1"]
    if encoder == "hevc_videotoolbox":
        return ["-c:v", "hevc_videotoolbox", "-b:v", bitrate, "-profile:v", "main10",
                "-pix_fmt", "p010le", "-tag:v", "hvc1"]
    if encoder == "libx265":
        return ["-c:v", "libx265", "-preset", "medium", "-b:v", bitrate, "-maxrate", peak, "-bufsize", buf,
                "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-x265-params", "log-level=error"]
    return ["-c:v", "libx264", "-preset", "medium", "-b:v", bitrate, "-pix_fmt", "yuv420p"]


AUDIO_ARGS = ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"]
COLOR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


def extract_frame(src: str | Path, t: float, dst: str | Path, vf: str | None = None) -> Path:
    args = ["-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1"]
    if vf:
        args += ["-vf", vf]
    run(args + ["-q:v", "2", str(dst)])
    return Path(dst)
