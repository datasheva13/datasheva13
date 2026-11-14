"""Filter-graph builders for the 9:16 blurred-background short and the 16:9 long video."""
from __future__ import annotations

from pathlib import Path

from .grade import ClipNormalization, Grade, filter_chain


def _esc_path(p: str | Path) -> str:
    """Escape a path for use inside an FFmpeg filter argument (Windows-safe)."""
    return str(Path(p).resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def vertical_graph(
    out_w: int,
    out_h: int,
    duration: float,
    grade: Grade,
    norm: ClipNormalization | None = None,
    push_in: float = 0.06,
    blur_sigma: float = 40,
    background_dim: float = 0.72,
    fps: float | None = None,
    hook_textfile: str | Path | None = None,
    font: str | Path | None = None,
    hook_seconds: float = 2.8,
) -> str:
    """16:9 source -> 9:16 frame: sharp graded footage centered on a blurred, darker copy.

    The foreground does a slow push-in (eval=frame scale + center crop) so a static
    chin-mount shot feels like a deliberate camera move. Output label: [vout].
    """
    # 2160 wide -> 1214 tall. Floor + 2px margin so the crop never exceeds the scaled
    # height after even-rounding (a 1-2 px overshoot makes FFmpeg fail).
    fg_h = int(out_w * 9 / 16 // 2 * 2) - 2
    work_w = int(round(out_w * (1 + push_in) / 2) * 2)      # spare pixels for the push-in
    chain = filter_chain(grade, norm, width=out_w)
    pre = f"fps={fps}," if fps else ""
    d = max(duration, 0.1)

    g = [
        f"[0:v]{pre}scale={work_w}:-2:flags=lanczos,{chain},split=2[fgsrc][bgsrc]",
        # background: cheap blur at quarter res, then upscale; dimmed so the subject pops
        f"[bgsrc]scale={out_w // 4}:{out_h // 4}:force_original_aspect_ratio=increase,"
        f"crop={out_w // 4}:{out_h // 4},gblur=sigma={blur_sigma / 4:.2f},"
        f"colorchannelmixer=rr={background_dim}:gg={background_dim}:bb={background_dim},"
        f"scale={out_w}:{out_h}:flags=bicubic,setsar=1[bg]",
        # foreground: slow push-in from 1.0x to (1+push_in)x of the output width
        f"[fgsrc]scale=w='trunc({out_w}*(1+{push_in}*t/{d:.3f})/2)*2':h=-2:eval=frame:flags=lanczos,"
        f"crop={out_w}:{fg_h},setsar=1[fg]",
        f"[bg][fg]overlay=x=0:y=(H-h)/2:shortest=1,format=yuv420p10le[vbase]",
    ]
    if hook_textfile and font:
        size = int(out_w * 0.058)
        y = int(out_h * 0.22)
        fade_in, fade_out = 0.35, 0.5
        alpha = (f"if(lt(t,{fade_in}),t/{fade_in},if(lt(t,{hook_seconds - fade_out}),1,"
                 f"max(0,({hook_seconds}-t)/{fade_out})))")
        g.append(
            f"[vbase]drawtext=fontfile='{_esc_path(font)}':textfile='{_esc_path(hook_textfile)}':"
            f"fontsize={size}:fontcolor=white:alpha='{alpha}':x=(w-text_w)/2:y={y}:"
            f"shadowcolor=black@0.45:shadowx=0:shadowy={max(2, size // 18)}:"
            f"enable='lt(t,{hook_seconds})'[vout]"
        )
    else:
        g.append("[vbase]null[vout]")
    return ";".join(g)


def horizontal_segment_chain(out_w: int, out_h: int, grade: Grade, norm: ClipNormalization | None,
                             fps: float | None) -> str:
    """Per-segment chain for the 16:9 long video (no labels)."""
    pre = f"fps={fps}," if fps else ""
    return (f"{pre}scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={out_w}:{out_h},{filter_chain(grade, norm, width=out_w)},setsar=1,format=yuv420p10le")


def audio_chain(duration: float, fade: float = 0.35, loudness: float = -14.0) -> str:
    """Raw exhaust audio: gentle fades, platform loudness (dynamics kept: LRA 11)."""
    out_start = max(duration - fade, 0)
    return (f"afade=t=in:st=0:d={fade},afade=t=out:st={out_start:.3f}:d={fade},"
            f"loudnorm=I={loudness}:TP=-1.0:LRA=11")
