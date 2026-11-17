"""Is this a motorcycle ride, and is some of the bike visible?

Once per source file, Claude looks at ~10 small frames spread through the video and
answers per frame: motorcycle or not (mountain bike, hiking, car...), and how much of
the bike is in view (good: tank/bars/dash/fender visible; bad: only windshield or none).
Cost: roughly $0.02 per video, cached in the database.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..captions.generate import _img_block

SCHEMA = {
    "type": "object",
    "properties": {
        "is_motorcycle_ride": {"type": "boolean"},
        "activity": {"type": "string", "description": "e.g. motorcycle, mountain bike, hiking, car, other"},
        "frames": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "bike_view": {"type": "string", "enum": ["bike_visible", "windshield_only", "no_bike"]},
                "colors": {"type": "integer", "description": "1-5: how beautiful/rich the natural colors are"},
            },
            "required": ["index", "bike_view", "colors"],
            "additionalProperties": False,
        }},
    },
    "required": ["is_motorcycle_ride", "activity", "frames"],
    "additionalProperties": False,
}

PROMPT = """These {n} frames come from one GoPro video (in time order, numbered from 0).
The channel only posts MOTORCYCLE rides (a CFMOTO Ibex 450 adventure bike, chin-mounted camera).
1. Is this a motorcycle ride? Mountain biking / bicycles, hiking, skiing, cars etc. = false.
2. For each frame, bike_view:
   - "bike_visible": part of the motorcycle is in view (handlebars, dash, tank, fender, mirrors, hands on bars)
   - "windshield_only": only the windshield/screen edge, nothing else of the bike
   - "no_bike": no part of the bike
3. For each frame, colors 1-5: 5 = rich, beautiful natural color (autumn, golden light, deep greens, blue sky),
   1 = flat/grey/washed out."""


def check_bike(proxy: str | Path, duration: float, frames_dir: Path, n: int = 10, model: str = "claude-opus-5-5") -> dict:
    """Returns {"is_motorcycle_ride", "activity", "times": [...], "views": [...], "colors": [...]} or {"error": ...}."""
    import anthropic

    from ..edit.render import sample_frames
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    times = [duration * (i + 0.5) / n for i in range(n)]
    paths = []
    for i, arr in enumerate(sample_frames(proxy, times, size=(320, 180))):
        p = frames_dir / f"check_{i}.jpg"
        Image.fromarray((arr * 255).astype(np.uint8)).save(p, quality=85)
        paths.append(p)
    if not paths:
        return {"error": "no frames"}
    content = [_img_block(p, 320) for p in paths] + [{"type": "text", "text": PROMPT.format(n=len(paths))}]
    try:
        resp = anthropic.Anthropic().beta.messages.create(
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    except anthropic.APIConnectionError:
        return {"error": "no connection"}
    except anthropic.APIStatusError as e:
        return {"error": f"API error {e.status_code}: {e.message}"}
    if resp.stop_reason == "refusal":
        return {"error": "declined"}
    try:
        data = json.loads("".join(b.text for b in resp.content if getattr(b, "type", "") == "text"))
    except json.JSONDecodeError:
        return {"error": "unreadable reply"}
    by_index = {f["index"]: f for f in data.get("frames", [])}
    views = [by_index.get(i, {}).get("bike_view", "bike_visible") for i in range(len(paths))]
    colors = [int(by_index.get(i, {}).get("colors", 3)) for i in range(len(paths))]
    return {"is_motorcycle_ride": bool(data.get("is_motorcycle_ride", True)), "activity": data.get("activity", ""),
            "times": times[:len(paths)], "views": views, "colors": colors}


def per_second_factors(check: dict | None, n_sec: int) -> tuple[np.ndarray, np.ndarray]:
    """(visibility multiplier, color score 0..1) per second, from the nearest checked frame."""
    vis = np.ones(n_sec)
    col = np.full(n_sec, 0.5)
    if not check or "times" not in check or not check["times"]:
        return vis, col
    times = np.array(check["times"])
    mult = {"bike_visible": 1.0, "windshield_only": 0.35, "no_bike": 0.15}
    for s in range(n_sec):
        i = int(np.argmin(np.abs(times - s)))
        vis[s] = mult.get(check["views"][i], 1.0)
        col[s] = (check["colors"][i] - 1) / 4
    return vis, col
