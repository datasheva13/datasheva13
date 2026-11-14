"""Captions, titles, descriptions, hashtags and thumbnail words via the Claude API.

Claude sees a few frames from the clip plus what the analysis measured (sound
intensity, speed, climbs), so hooks match what's actually on screen.
"""
from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = {
    "type": "object",
    "properties": {
        "hooks": {"type": "array", "items": {"type": "string"},
                  "description": "3 short on-screen/caption hooks"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"},
                     "description": "exactly 3: one big, one mid, one niche"},
        "thumbnail_words": {"type": "array", "items": {"type": "string"},
                            "description": "3 single-word thumbnail options, uppercase"},
    },
    "required": ["hooks", "title", "description", "hashtags", "thumbnail_words"],
    "additionalProperties": False,
}

SYSTEM = """You write captions for a raw-sound, no-talking, off-road motorcycle POV channel.
Bike: {bike}. Footage: GoPro chin mount, cinematic grade, real exhaust sound, no music by default.

Voice: {tone}

Rules:
- Hooks: 2-7 words. Moody, slightly edgy, confident, a little mysterious. They must make a
  scroller stop and turn the sound up. Vary the structure; never reuse a recent hook.
  Good: "Some trails don't want to be found." / "Headphones. Now." / "Nobody out here but the engine."
  Bad: "Check out this AMAZING ride!!!" / "You won't BELIEVE..." / anything with more than one emoji.
- Title (YouTube): under 70 characters, intriguing first, then searchable keywords
  (bike name, "raw sound", "off-road POV"). For shorts end with nothing extra; YouTube detects Shorts.
- Description: 1-2 moody lines, then one plain line with searchable keywords
  (e.g. "CFMOTO Ibex 450 raw exhaust sound, off-road trail POV, no music"). No hashtags in it.
- Hashtags: exactly 3, pick one from each list: big {big}; mid {mid}; niche {niche}.
- Thumbnail words: 3 options, ONE word each, uppercase, evocative and cinematic
  (e.g. ASCENT, ALONE, FERAL, DUSK, RIDGE, SILENCE). Match the scenery and mood."""


@dataclass
class CaptionSet:
    hooks: list[str]
    title: str
    description: str
    hashtags: list[str]
    thumbnail_words: list[str]
    raw: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"hooks": self.hooks, "title": self.title, "description": self.description,
                "hashtags": self.hashtags, "thumbnail_words": self.thumbnail_words}


def _img_block(path: Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(path.read_bytes()).decode()}}


def _fix_hashtags(tags: list[str], pool: dict) -> list[str]:
    tags = [t if t.startswith("#") else f"#{t}" for t in tags]
    tags = [t.replace(" ", "").lower() for t in tags]
    out = []
    for group in ("big", "mid", "niche"):
        options = [p.lower() for p in pool[group]]
        chosen = next((t for t in tags if t in options and t not in out), None)
        out.append(chosen or random.choice(options))
    return out


def fallback_captions(pool: dict, kind: str = "short") -> CaptionSet:
    """Used when no API key is set or the call fails, so the pipeline never blocks."""
    hooks = random.sample(["Headphones. Now.", "Nobody out here but the engine.",
                           "Some trails don't want to be found.", "Turn it up.", "No music. Just this.",
                           "The quiet part ends here."], 3)
    return CaptionSet(hooks, "Raw sound | CFMOTO Ibex 450 off-road POV",
                      "Nothing but the trail and the twin.\nCFMOTO Ibex 450 raw exhaust sound, off-road POV, no music.",
                      _fix_hashtags([], pool), ["ASCENT", "ALONE", "DUSK"])


def generate_captions(cfg, frames: list[Path], stats: dict, kind: str = "short",
                      recent_hooks: list[str] | None = None) -> CaptionSet:
    import anthropic

    ccfg = cfg["captions"]
    pool = ccfg["hashtag_pool"]
    system = SYSTEM.format(bike=ccfg["bike"], tone=ccfg["tone"].strip(),
                           big=", ".join(pool["big"]), mid=", ".join(pool["mid"]), niche=", ".join(pool["niche"]))
    what = "a 9:16 short (7-20 s)" if kind == "short" else "a long horizontal YouTube ride video (several minutes)"
    text = (f"Write captions for {what}. Frames from the clip are attached.\n"
            f"Measured: {json.dumps(stats)}\n"
            f"Recent hooks to avoid repeating: {json.dumps((recent_hooks or [])[-15:])}")
    content = [_img_block(p) for p in frames[:4] if p.exists()] + [{"type": "text", "text": text}]

    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=ccfg.get("model", "claude-opus-5-5"),
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APIConnectionError as e:
        print(f"[captions] connection problem, using fallback captions: {e}")
        return fallback_captions(pool, kind)
    except anthropic.RateLimitError as e:
        print(f"[captions] rate limited, using fallback captions: {e}")
        return fallback_captions(pool, kind)
    except anthropic.APIStatusError as e:
        print(f"[captions] API error {e.status_code}, using fallback captions: {e.message}")
        return fallback_captions(pool, kind)

    if resp.stop_reason == "refusal":
        print("[captions] request declined, using fallback captions")
        return fallback_captions(pool, kind)
    raw_text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return fallback_captions(pool, kind)
    hooks = [h.strip() for h in data.get("hooks", []) if h.strip()][:3] or fallback_captions(pool).hooks
    words = [w.strip().split()[0].upper() for w in data.get("thumbnail_words", []) if w.strip()][:3]
    return CaptionSet(hooks, data.get("title", "")[:100], data.get("description", ""),
                      _fix_hashtags(data.get("hashtags", []), pool), words or ["ASCENT", "ALONE", "DUSK"], data)


def caption_text(hook: str, description: str, hashtags: list[str]) -> str:
    """The text you paste into TikTok / Instagram."""
    return f"{hook}\n\n{description}\n\n{' '.join(hashtags)}".strip()
