"""Captions, titles, descriptions, hashtags and thumbnail words via the Claude API.

Claude sees frames from the clip plus what the analysis measured (sound intensity,
speed, climbs, light), and the history of what was already used, so every video gets
its own hooks, title, description, hashtags and thumbnail words.
"""
from __future__ import annotations

import base64
import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

N_HOOKS = 5
N_WORDS = 5

SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string", "description": "one sentence: what is actually in the frames"},
        "hooks": {"type": "array", "items": {"type": "string"}, "description": f"{N_HOOKS} hooks"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"},
                     "description": "exactly 3: one big, one mid, one niche"},
        "thumbnail_words": {"type": "array", "items": {"type": "string"},
                            "description": f"{N_WORDS} single uppercase words"},
    },
    "required": ["scene", "hooks", "title", "description", "hashtags", "thumbnail_words"],
    "additionalProperties": False,
}

SYSTEM = """You write for a raw-sound, no-talking, off-road motorcycle POV channel.
Bike: {bike}. Footage: GoPro chin mount, cinematic grade, real exhaust sound, no voice.

Voice: {tone}

First look closely at the frames: terrain, light, weather, season, colors, what the
trail does next. Everything you write must come from THIS clip, not a template.

HOOKS ({n_hooks}): on-screen text for the first 3 seconds, max 8 words.
- Tell a tiny story or carry an emotion: solitude, escape, obsession, the last ride
  before winter, a place nobody else goes, what the rider is running from or toward.
- Make the viewer feel something or need to see what happens next. Specific beats generic.
- Each of the {n_hooks} must use a different angle (story / feeling / tension / place / truth).
- Bad (too basic): "Turn it up." "Headphones on." "Pure sound." "Enjoy the ride."
- Better: "Nobody followed me past this point." / "I ride here so I don't have to talk." /
  "The trail ended. The throttle didn't." / "Last light. Last climb. No signal."
- No hashtags, max one emoji across all hooks, never clickbait or ALL CAPS.

TITLE (YouTube): under 70 characters. Lead with intrigue drawn from this clip, then
searchable words (CFMOTO Ibex 450 / 450MT, raw sound, off-road POV). Unique every time.

DESCRIPTION: 2 moody lines specific to this clip + 1 plain keyword line
(e.g. "CFMOTO Ibex 450 raw exhaust sound, off-road trail POV, no music"). No hashtags.

HASHTAGS: exactly 3, one from each list, choose the best fit for this clip:
big {big}; mid {mid}; niche {niche}.

THUMBNAIL WORDS ({n_words}): one word each, uppercase, cinematic, evocative, drawn from
this clip's scenery, light and mood (e.g. a ridge at dusk -> EMBER, RIDGELINE, HOLLOW...).
All {n_words} different in feeling. Never use any word from the "already used" list."""


@dataclass
class CaptionSet:
    hooks: list[str]
    title: str
    description: str
    hashtags: list[str]
    thumbnail_words: list[str]
    note: str = ""                 # empty = written by Claude; otherwise why placeholders were used
    raw: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"hooks": self.hooks, "title": self.title, "description": self.description,
                "hashtags": self.hashtags, "thumbnail_words": self.thumbnail_words, "note": self.note,
                "scene": self.raw.get("scene", "")}


def _img_block(path: Path, max_side: int = 1024) -> dict:
    """Downscale frames before sending: 4K JPEGs can exceed the API's 5 MB image limit."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


def _fix_hashtags(tags: list[str], pool: dict) -> list[str]:
    tags = [t if t.startswith("#") else f"#{t}" for t in tags]
    tags = [t.replace(" ", "").lower() for t in tags]
    out = []
    for group in ("big", "mid", "niche"):
        options = [p.lower() for p in pool[group]]
        chosen = next((t for t in tags if t in options and t not in out), None)
        out.append(chosen or random.choice(options))
    return out


_FB_HOOKS = ["Nobody followed me past this point.", "I ride here so I don't have to talk.",
             "The trail ended. The throttle didn't.", "Last light. Last climb. No signal.",
             "Some places you only reach alone.", "This is where the noise in my head stops.",
             "Winter's coming. One more run.", "Didn't plan to come back this late.",
             "The map said no road here.", "Every weekend, the same escape.",
             "Two cylinders and nowhere to be.", "The quiet up here is loud."]
_FB_WORDS = ["ASCENT", "HOLLOW", "EMBER", "RIDGELINE", "DRIFT", "SOLACE", "WILD", "RELIC", "TIMBER",
             "GRIT", "NORTH", "VEIL", "EXILE", "STONE", "HAZE", "FERAL", "DUSK", "ALONE"]


def fallback_captions(pool: dict, kind: str = "short", reason: str = "no API key",
                      avoid_words: list[str] | None = None, avoid_hooks: list[str] | None = None) -> CaptionSet:
    """Used when the API call fails so the pipeline never blocks. Clearly flagged in the dashboard."""
    avoid = {w.upper() for w in (avoid_words or [])}
    words = [w for w in _FB_WORDS if w not in avoid] or _FB_WORDS
    title = random.choice(["Raw sound", "No music, no talking", "Just the twin", "Off the map"])
    used = {h.lower() for h in (avoid_hooks or [])}
    hooks = [h for h in _FB_HOOKS if h.lower() not in used]
    hooks = hooks if len(hooks) >= N_HOOKS else _FB_HOOKS
    return CaptionSet(random.sample(hooks, N_HOOKS), f"{title} | CFMOTO Ibex 450 off-road POV",
                      "Nothing but the trail and the twin.\nCFMOTO Ibex 450 raw exhaust sound, off-road POV, no music.",
                      _fix_hashtags([], pool), random.sample(words, min(N_WORDS, len(words))),
                      note=f"Placeholder captions ({reason}). Press 'New caption ideas' to retry.")


def generate_captions(cfg, frames: list[Path], stats: dict, kind: str = "short",
                      history: dict | None = None) -> CaptionSet:
    """history: {"hooks": [...], "titles": [...], "words_used": [...]} to avoid repeats."""
    import anthropic

    ccfg = cfg["captions"]
    pool = ccfg["hashtag_pool"]
    history = history or {}
    system = SYSTEM.format(bike=ccfg["bike"], tone=ccfg["tone"].strip(), n_hooks=N_HOOKS, n_words=N_WORDS,
                           big=", ".join(pool["big"]), mid=", ".join(pool["mid"]), niche=", ".join(pool["niche"]))
    what = "a 9:16 short (10-20 s, it loops)" if kind == "short" else "a long horizontal YouTube ride video"
    text = (f"Write everything for {what}. {len(frames[:4])} frames from the clip are attached.\n"
            f"Measured: {json.dumps(stats)}\n"
            f"Hooks already used (never repeat or paraphrase): {json.dumps(history.get('hooks', [])[:40])}\n"
            f"Titles already used (don't repeat): {json.dumps(history.get('titles', [])[:25])}\n"
            f"Thumbnail words already used (never use): {json.dumps(history.get('words_used', [])[:80])}")
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
    except anthropic.AuthenticationError:
        return fallback_captions(pool, kind, "API key rejected - check .env", history.get("words_used"), history.get("hooks"))
    except anthropic.PermissionDeniedError as e:
        return fallback_captions(pool, kind, f"permission denied: {e.message}", history.get("words_used"), history.get("hooks"))
    except anthropic.RateLimitError:
        return fallback_captions(pool, kind, "rate limited", history.get("words_used"), history.get("hooks"))
    except anthropic.APIStatusError as e:
        return fallback_captions(pool, kind, f"API error {e.status_code}: {e.message}", history.get("words_used"), history.get("hooks"))
    except anthropic.APIConnectionError:
        return fallback_captions(pool, kind, "no connection to the API", history.get("words_used"), history.get("hooks"))

    if resp.stop_reason == "refusal":
        return fallback_captions(pool, kind, "request declined", history.get("words_used"), history.get("hooks"))
    raw_text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return fallback_captions(pool, kind, "unreadable reply", history.get("words_used"), history.get("hooks"))

    used_hooks = {h.strip().lower() for h in history.get("hooks", [])}
    hooks = [h.strip() for h in data.get("hooks", []) if h.strip() and h.strip().lower() not in used_hooks][:N_HOOKS]
    used_words = {w.upper() for w in history.get("words_used", [])}
    words = []
    for w in data.get("thumbnail_words", []):
        w = w.strip().split()[0].upper() if w.strip() else ""
        if w and w not in used_words and w not in words:
            words.append(w)
    fb = fallback_captions(pool, kind, "", history.get("words_used"), history.get("hooks"))
    return CaptionSet(hooks or fb.hooks, (data.get("title") or fb.title)[:100], data.get("description") or fb.description,
                      _fix_hashtags(data.get("hashtags", []), pool), (words or fb.thumbnail_words)[:N_WORDS],
                      note="", raw=data)


def caption_text(hook: str, description: str, hashtags: list[str]) -> str:
    """The text you paste into TikTok / Instagram."""
    return f"{hook}\n\n{description}\n\n{' '.join(hashtags)}".strip()
