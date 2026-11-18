"""Combine per-second scores into ranked clip windows, with diversity across days."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import AudioFeatures
from .telemetry import Telemetry, telemetry_score
from .video import VideoFeatures

W_AUDIO, W_SCENE, W_TELEM = 0.50, 0.35, 0.15


@dataclass
class Analysis:
    """Everything we know about one source file, per second."""
    duration: float
    total: np.ndarray
    audio: np.ndarray
    scene: np.ndarray
    telem: np.ndarray
    quality: np.ndarray
    embeddings: np.ndarray
    onset_times: np.ndarray


@dataclass
class Candidate:
    source_id: int
    start: float
    duration: float
    score: float
    audio: float
    scene: float
    embedding: np.ndarray = field(repr=False)
    stats: dict = field(default_factory=dict)
    split: float | None = None        # loop: the short opens here (the best moment)

    @property
    def end(self) -> float:
        return self.start + self.duration


def combine(a: AudioFeatures, v: VideoFeatures, t: Telemetry, duration: float) -> Analysis:
    n = max(1, min(len(a.score), len(v.score)))
    au, sc = a.score[:n], v.score[:n]
    te = telemetry_score(t, n)
    if t.empty:
        total = (W_AUDIO * au + W_SCENE * sc) / (W_AUDIO + W_SCENE)
    else:
        total = W_AUDIO * au + W_SCENE * sc + W_TELEM * te
    total = total * np.clip(v.quality[:n] * 1.5, 0, 1)     # unusable frames sink
    return Analysis(duration, total, au, sc, te, v.quality[:n], v.embeddings[:n], a.onset_times)


def _snap_to_onset(t: float, onsets: np.ndarray, lead: float = 0.6, window: float = 1.5) -> float:
    """Start a hair before a throttle hit so the short opens on sound."""
    if len(onsets) == 0:
        return t
    near = onsets[(onsets >= t) & (onsets <= t + window)]
    return max(float(near[0]) - lead, 0.0) if len(near) else t


def windows(an: Analysis, source_id: int, min_len: int, max_len: int, step: int = 1,
            loop: bool = False) -> list[Candidate]:
    """Candidate clips. loop=True scores them for the loop layout (attention first):
    the short opens at the split, so the best moment must sit right after it."""
    n = len(an.total)
    out: list[Candidate] = []
    if n < min_len:
        return out
    csum = np.concatenate([[0], np.cumsum(an.total)])
    for length in sorted({min_len, (min_len + max_len) // 2, max_len}):
        if length > n:
            continue
        for s in range(0, n - length + 1, step):
            if an.quality[s:s + length].min() < 0.3:
                continue
            seg = an.total[s:s + length]
            dip = float(np.percentile(seg, 15))                   # no dead patches
            # raw-sound shorts breathe better at 12-18 s than at the minimum length
            length_bonus = 0.06 * (length - min_len) / max(max_len - min_len, 1)
            split = None
            if loop:
                score, split = _loop_score(an, s, length, dip)
                score += length_bonus
            else:
                mean = (csum[s + length] - csum[s]) / length
                hook = float(an.total[s:s + 2].mean())           # strong first 2 seconds
                score = 0.55 * mean + 0.25 * hook + 0.20 * dip + length_bonus
            emb = an.embeddings[s:s + length].mean(axis=0)
            emb = emb / max(np.linalg.norm(emb), 1e-6)
            out.append(Candidate(source_id, float(s), float(length), float(score),
                                 float(an.audio[s:s + length].mean()), float(an.scene[s:s + length].mean()), emb,
                                 split=split))
    return out


def _loop_score(an: Analysis, s: int, length: int, dip: float) -> tuple[float, float]:
    """Priority: opening punch 50% > rest stays engaging 30% > smooth internal cut 20%."""
    e = s + length
    lo, hi = s + int(np.ceil(length * 0.40)), s + int(np.floor(length * 0.65))
    best_p, best_open = lo, -1.0
    for p in range(lo, max(hi, lo) + 1):
        opening = float(an.total[p:min(p + 3, e)].mean())
        if opening > best_open:
            best_open, best_p = opening, p
    rest = np.concatenate([an.total[s:best_p], an.total[min(best_p + 3, e):e]])
    rest_score = 0.7 * float(rest.mean() if len(rest) else 0) + 0.3 * dip
    # smoothness estimate: the clip's last second should look and sound like its first
    look = float(an.embeddings[e - 1] @ an.embeddings[s]) if len(an.embeddings) >= e else 0.5
    sound = 1.0 - min(abs(float(an.audio[e - 1]) - float(an.audio[s])), 1.0)
    smooth = 0.5 * np.clip((look - 0.8) / 0.2, 0, 1) + 0.5 * sound
    return 0.5 * best_open + 0.3 * rest_score + 0.2 * float(smooth), float(best_p)


def select(cands: list[Candidate], k: int, history: list[np.ndarray] | None = None,
           similarity_limit: float = 0.93, onsets: dict[int, np.ndarray] | None = None) -> list[Candidate]:
    """Greedy pick: best score first, no overlap, and not too similar to picks or recent history."""
    history = history or []
    picked: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if len(picked) >= k:
            break
        if any(p.source_id == c.source_id and c.start < p.end + 3 and p.start < c.end + 3 for p in picked):
            continue
        refs = [p.embedding for p in picked] + history
        if refs and max(float(r @ c.embedding) for r in refs) > similarity_limit:
            continue
        picked.append(c)
    # relax diversity if footage was too uniform to fill the quota
    if len(picked) < k and similarity_limit < 0.995:
        return select(cands, k, history, similarity_limit + 0.02, onsets)
    if onsets:
        for c in picked:
            ons = onsets.get(c.source_id, np.array([]))
            if c.split is not None:            # open exactly on the throttle hit nearest the peak
                near = ons[(ons >= c.split - 0.8) & (ons <= c.split + 0.8)]
                if len(near):
                    c.split = max(float(near[np.argmin(np.abs(near - c.split))]) - 0.12, c.start + 1.5)
            else:
                c.start = _snap_to_onset(c.start, ons)
    return picked
