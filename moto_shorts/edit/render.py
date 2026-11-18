"""Rendering: shorts (9:16 masters + 1080p social), long videos (16:9), clip normalization."""
from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import ffmpeg as ff
from .grade import ClipNormalization, Grade, measure_normalization
from .reframe import audio_chain, horizontal_segment_chain, rotated_graph, vertical_graph


def sample_frames(src: str | Path, times: list[float], size: tuple[int, int] = (192, 108)) -> list[np.ndarray]:
    """Grab small RGB frames (float 0..1) at the given times."""
    frames = []
    w, h = size
    for t in times:
        proc = subprocess.run(
            [ff.FFMPEG, "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(src), "-frames:v", "1",
             "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True,
        )
        if len(proc.stdout) == w * h * 3:
            frames.append(np.frombuffer(proc.stdout, np.uint8).reshape(h, w, 3).astype(np.float32) / 255)
    return frames


def normalization_for(src: str | Path, start: float, duration: float, grade: Grade) -> ClipNormalization:
    times = [start + duration * k / 5 for k in range(1, 5)]
    return measure_normalization(sample_frames(src, times), grade)


@dataclass
class RenderSettings:
    encoder: str
    bitrate_master: str = "100M"
    bitrate_social: str = "25M"
    fps: float | None = None              # None = keep source
    short_size: tuple[int, int] = (2160, 3840)
    long_size: tuple[int, int] = (3840, 2160)
    social_size: tuple[int, int] = (1080, 1920)
    push_in: float = 0.06
    blur_sigma: float = 40
    background_dim: float = 0.72
    audio_fade: float = 0.35
    music_volume_db: float = -16
    loop_crossfade: float = 0.25          # blend at the loop's internal cut (0 = hard cut)

    @classmethod
    def from_config(cls, cfg) -> "RenderSettings":
        r, s = cfg["render"], cfg["style"]
        fps = None if str(r.get("fps", "source")) == "source" else float(r["fps"])
        return cls(
            encoder=ff.pick_encoder(r.get("encoder", "auto")),
            bitrate_master=r.get("bitrate_master", "100M"),
            bitrate_social=r.get("bitrate_social", "25M"),
            fps=fps,
            short_size=tuple(r.get("short_size", (2160, 3840))),
            long_size=tuple(r.get("long_size", (3840, 2160))),
            social_size=tuple(r.get("social_size", (1080, 1920))),
            push_in=float(s.get("push_in", 0.06)),
            blur_sigma=float(s.get("blur_sigma", 40)),
            background_dim=float(s.get("background_dim", 0.72)),
            audio_fade=float(s.get("audio_fade", 0.35)),
            music_volume_db=float(s.get("music_volume_db", -16)),
            loop_crossfade=float(s.get("loop_crossfade", 0.25)),
        )


def _music_inputs(music: Path | None, duration: float) -> tuple[list[str], str | None]:
    if not music:
        return [], None
    info = ff.probe(music)
    offset = random.uniform(0, max(info.duration - duration - 1, 0)) if info.duration > duration + 2 else 0
    return ["-ss", f"{offset:.2f}", "-t", f"{duration + 0.5:.2f}", "-i", str(music)], str(music)


def render_short(
    src: str | Path,
    start: float,
    duration: float,
    out_master: str | Path,
    settings: RenderSettings,
    grade: Grade,
    norm: ClipNormalization | None = None,
    hook_textfile: str | Path | None = None,
    font: str | Path | None = None,
    music: Path | None = None,
    loop_split: float | None = None,
) -> Path:
    """Render the 9:16 YouTube master (4K vertical, 10-bit HEVC, ~100 Mbps).

    loop_split (seconds into the source, inside the clip): play [split, end] first, then
    cut to [start, split]. The video ends right where it began, so replays feel endless.
    """
    out_w, out_h = settings.short_size
    end = start + duration
    if loop_split is not None and start + 1.0 < loop_split < end - 1.0:
        b_len, a_len = end - loop_split, loop_split - start
        # blend length: limited by the footage available after `end` and by the part lengths
        x = max(0.0, min(settings.loop_crossfade, a_len / 3, b_len / 3,
                         max(ff.probe(src).duration - end - 0.05, 0.0)))
        if x >= 0.04:
            inputs = ["-ss", f"{loop_split:.3f}", "-t", f"{b_len + x:.3f}", "-i", str(src),
                      "-ss", f"{start:.3f}", "-t", f"{a_len:.3f}", "-i", str(src)]
            # soft dissolve + equal-power audio crossfade at the internal cut (length unchanged)
            pre = (f"[0:v]settb=AVTB,setpts=PTS-STARTPTS[v0];[1:v]settb=AVTB,setpts=PTS-STARTPTS[v1];"
                   f"[v0][v1]xfade=transition=fade:duration={x:.3f}:offset={b_len:.3f}[cv];"
                   f"[0:a]asetpts=PTS-STARTPTS[a0];[1:a]asetpts=PTS-STARTPTS[a1];"
                   f"[a0][a1]acrossfade=d={x:.3f}:c1=qsin:c2=qsin[ca];")
        else:
            inputs = ["-ss", f"{loop_split:.3f}", "-t", f"{b_len:.3f}", "-i", str(src),
                      "-ss", f"{start:.3f}", "-t", f"{a_len:.3f}", "-i", str(src)]
            # hard cut: tiny audio fades so the jump doesn't click
            pre = (f"[0:a]afade=t=out:st={max(b_len - 0.04, 0):.3f}:d=0.04[ab];"
                   f"[1:a]afade=t=in:d=0.04[aa];"
                   f"[0:v][ab][1:v][aa]concat=n=2:v=1:a=1[cv][ca];")
        vin, ain, music_idx = "[cv]", "[ca]", 2
    else:
        inputs = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src)]
        pre, vin, ain, music_idx = "", "[0:v]", "[0:a]", 1
    vgraph = vertical_graph(out_w, out_h, duration, grade, norm, settings.push_in, settings.blur_sigma,
                            settings.background_dim, settings.fps, hook_textfile, font, vin=vin)
    music_args, music_path = _music_inputs(music, duration)
    # loop: no fade-in/out on the outer ends (they join seamlessly on replay)
    a = audio_chain(duration, 0.0 if pre else settings.audio_fade)
    if music_path:
        mfade = max(duration - 1.2, 0)
        agraph = (f"{ain}{a}[ex];[{music_idx}:a]volume={settings.music_volume_db}dB,afade=t=in:d=1.0,"
                  f"afade=t=out:st={mfade:.2f}:d=1.2[mu];[ex][mu]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        agraph = f"{ain}{a}[aout]"
    args = [*inputs, *music_args,
            "-filter_complex", f"{pre}{vgraph};{agraph}", "-map", "[vout]", "-map", "[aout]",
            *ff.video_codec_args(settings.encoder, settings.bitrate_master), *ff.COLOR_TAGS, *ff.AUDIO_ARGS,
            "-movflags", "+faststart", str(out_master)]
    ff.run(args)
    return Path(out_master)


def render_social(master: str | Path, out: str | Path, settings: RenderSettings) -> Path:
    """1080p export for Instagram/TikTok from the master (our lanczos downscale beats theirs).
    8-bit H.264 here: both apps handle it most predictably."""
    w, h = settings.social_size
    ff.run(["-i", str(master), "-vf", f"scale={w}:{h}:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-b:v", settings.bitrate_social,
            "-maxrate", settings.bitrate_social, "-bufsize", "50M", "-profile:v", "high",
            *ff.COLOR_TAGS, "-c:a", "copy", "-movflags", "+faststart", str(out)])
    return Path(out)


def render_instagram_rotated(src: str | Path, start: float, duration: float, out: str | Path,
                             settings: RenderSettings, grade: Grade, norm: ClipNormalization | None = None,
                             hook_textfile: str | Path | None = None, font: str | Path | None = None,
                             direction: str = "clockwise", music: Path | None = None) -> Path:
    """Full-detail 1920x1080 landscape frame rotated into a 1080x1920 file (viewer turns the phone)."""
    sw, sh = settings.social_size                      # 1080 x 1920 -> landscape 1920 x 1080
    vgraph = rotated_graph(sh, sw, grade, norm, settings.fps, direction, hook_textfile, font)
    music_args, music_path = _music_inputs(music, duration)
    a = audio_chain(duration, settings.audio_fade)
    if music_path:
        mfade = max(duration - 1.2, 0)
        agraph = (f"[0:a]{a}[ex];[1:a]volume={settings.music_volume_db}dB,afade=t=in:d=1.0,"
                  f"afade=t=out:st={mfade:.2f}:d=1.2[mu];[ex][mu]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        agraph = f"[0:a]{a}[aout]"
    ff.run(["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src), *music_args,
            "-filter_complex", f"{vgraph};{agraph}", "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "slow", "-b:v", settings.bitrate_social,
            "-maxrate", settings.bitrate_social, "-bufsize", "50M", "-profile:v", "high",
            *ff.COLOR_TAGS, *ff.AUDIO_ARGS, "-movflags", "+faststart", str(out)])
    return Path(out)


@dataclass
class Segment:
    src: Path
    start: float
    duration: float
    norm: ClipNormalization | None = None
    title: str = ""


def render_long(segments: list[Segment], out: str | Path, settings: RenderSettings, grade: Grade,
                fades: bool = True) -> tuple[Path, list[tuple[float, str]]]:
    """16:9 long video: graded segments joined with tiny audio fades (no flashy transitions).
    Returns the output path and YouTube chapters [(start_seconds, title)]."""
    w, h = settings.long_size
    args: list[str] = []
    parts: list[str] = []
    chapters: list[tuple[float, str]] = []
    t = 0.0
    for i, seg in enumerate(segments):
        args += ["-ss", f"{seg.start:.3f}", "-t", f"{seg.duration:.3f}", "-i", str(seg.src)]
        chain = horizontal_segment_chain(w, h, grade, seg.norm, settings.fps)
        out_start = max(seg.duration - 0.08, 0)
        parts.append(f"[{i}:v]{chain}[v{i}]")
        parts.append(f"[{i}:a]afade=t=in:d=0.08,afade=t=out:st={out_start:.3f}:d=0.08,aresample=48000[a{i}]")
        chapters.append((t, seg.title or f"Part {i + 1}"))
        t += seg.duration
    n = len(segments)
    concat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
    parts.append(f"{concat_in}concat=n={n}:v=1:a=1[vc][ac]")
    if fades:
        parts.append(f"[vc]fade=t=in:d=1.0,fade=t=out:st={max(t - 1.5, 0):.3f}:d=1.5[vout]")
    else:                                   # your own edit: colour grade only, nothing added
        parts.append("[vc]null[vout]")
    parts.append("[ac]loudnorm=I=-14:TP=-1.0:LRA=11[aout]")
    ff.run(args + ["-filter_complex", ";".join(parts), "-map", "[vout]", "-map", "[aout]",
                   *ff.video_codec_args(settings.encoder, settings.bitrate_master), *ff.COLOR_TAGS,
                   *ff.AUDIO_ARGS, "-movflags", "+faststart", str(out)])
    return Path(out), chapters
