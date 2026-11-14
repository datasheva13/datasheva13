"""Low-res analysis proxies: a tiny 2 fps video + mono audio. Analysis never touches the 5.3K file."""
from __future__ import annotations

from pathlib import Path

from .. import ffmpeg as ff

PROXY_FPS = 2
PROXY_WIDTH = 320
AUDIO_SR = 22050


def make_proxies(src: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{src.stem}_proxy.mp4"
    audio = out_dir / f"{src.stem}_audio.wav"
    if video.exists() and audio.exists():
        return video, audio
    ff.run(["-hwaccel", "auto", "-i", str(src),
            "-map", "0:v:0", "-vf", f"fps={PROXY_FPS},scale={PROXY_WIDTH}:-2", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "28", "-an", str(video),
            "-map", "0:a:0?", "-ac", "1", "-ar", str(AUDIO_SR), "-c:a", "pcm_s16le", str(audio)])
    return video, audio
