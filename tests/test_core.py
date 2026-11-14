import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from moto_shorts.analyze.score import Analysis, Candidate, select, windows
from moto_shorts.analyze.telemetry import parse_gps5
from moto_shorts.captions.generate import _fix_hashtags, caption_text
from moto_shorts.db import DB
from moto_shorts.edit.grade import ClipNormalization, Grade, filter_chain, measure_normalization, write_cube_lut
from moto_shorts.edit.reframe import vertical_graph

POOL = {"big": ["#motorcycle", "#offroad"], "mid": ["#cfmoto", "#enduro"], "niche": ["#ibex450", "#450mt"]}


# ---------------------------------------------------------------- grade
def test_grade_strength_zero_is_neutral():
    g = Grade().with_strength(0)
    chain = filter_chain(g)
    assert "colorbalance" not in chain and "noise" not in chain and "vignette" not in chain
    assert "eq=contrast=1.0000:saturation=1.0000" in chain


def test_grade_full_has_all_stages():
    chain = filter_chain(Grade(), ClipNormalization(0.3, (1.05, 1.0, 0.95)))
    for stage in ("colorchannelmixer", "exposure", "curves", "colorbalance", "huesaturation", "noise", "vignette"):
        assert stage in chain


def test_normalization_corrects_blue_cast_gently():
    frame = np.full((50, 50, 3), [0.40, 0.45, 0.60], np.float32)
    n = measure_normalization([frame], Grade())
    r, g, b = n.gains_rgb
    assert r > 1.0 and b < 1.0
    assert b >= 0.92 - 1e-6 and r <= 1.08 + 1e-6     # gentle, never extreme


def test_normalization_exposure_clamped():
    dark = np.full((50, 50, 3), 0.05, np.float32)
    assert measure_normalization([dark], Grade()).exposure_ev == pytest.approx(Grade().max_exposure_ev)


def test_cube_lut(tmp_path):
    p = write_cube_lut(Grade(), tmp_path / "x.cube", size=5)
    lines = p.read_text().splitlines()
    assert lines[1] == "LUT_3D_SIZE 5" and len(lines) == 2 + 125


# ---------------------------------------------------------------- scoring
def _analysis(total):
    n = len(total)
    emb = np.eye(n, 8)[:, :8] + 0.01
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return Analysis(float(n), np.array(total), np.array(total), np.array(total), np.zeros(n), np.ones(n),
                    emb, np.array([12.3]))


def test_windows_find_the_loud_part():
    total = [0.1] * 40 + [0.9] * 15 + [0.1] * 40
    cands = windows(_analysis(total), 1, 9, 20)
    best = max(cands, key=lambda c: c.score)
    assert 38 <= best.start <= 46


def test_select_no_overlap_and_snaps_to_onset():
    total = [0.1] * 10 + [0.9] * 15 + [0.1] * 10 + [0.8] * 15 + [0.1] * 10
    an = _analysis(total)
    picks = select(windows(an, 1, 9, 12), 2, onsets={1: np.array([11.0, 36.5])})
    assert len(picks) == 2
    a, b = sorted(picks, key=lambda c: c.start)
    assert a.end <= b.start


def test_select_diversity_against_history():
    e = np.zeros(8, np.float32)
    e[0] = 1
    c1 = Candidate(1, 0, 10, 0.9, 0.9, 0.9, e)
    other = np.zeros(8, np.float32)
    other[1] = 1
    c2 = Candidate(2, 0, 10, 0.5, 0.5, 0.5, other)
    picks = select([c1, c2], 1, history=[e])
    assert picks[0].source_id == 2


# ---------------------------------------------------------------- telemetry
def _klv(key, typ, size, repeat, payload):
    pad = (-len(payload)) % 4
    return key.encode() + bytes([ord(typ), size]) + struct.pack(">H", repeat) + payload + b"\0" * pad


def test_gpmf_gps5_parse():
    scal = _klv("SCAL", "l", 4, 5, struct.pack(">5i", 10000000, 10000000, 1000, 1000, 100))
    rows = [(415000000, -724000000, 350000, 10000, 1000), (415000100, -724000100, 351000, 12000, 1200)]
    gps = _klv("GPS5", "l", 20, 2, b"".join(struct.pack(">5i", *r) for r in rows))
    strm = _klv("STRM", "\0", 1, len(scal + gps), scal + gps)
    devc = _klv("DEVC", "\0", 1, len(strm), strm)
    blocks = parse_gps5(devc)
    assert len(blocks) == 1
    assert blocks[0][0, 2] == pytest.approx(350.0)       # altitude m
    assert blocks[0][1, 3] == pytest.approx(12.0)        # speed m/s


# ---------------------------------------------------------------- captions / db
def test_hashtags_always_three_from_pool():
    tags = _fix_hashtags(["offroad", "#CFMOTO", "#random"], POOL)
    assert tags[0] == "#offroad" and tags[1] == "#cfmoto" and tags[2] in POOL["niche"]
    assert len(_fix_hashtags([], POOL)) == 3


def test_caption_text():
    assert caption_text("Hook.", "Desc", ["#a", "#b", "#c"]) == "Hook.\n\nDesc\n\n#a #b #c"


def test_db_used_ranges(tmp_path):
    db = DB(tmp_path / "t.db")
    sid = db.add_source("x", "f.mp4", "/f.mp4", None, 100)
    c = db.add_clip("short", sid, 10, 15, 0.5, np.ones(4))
    db.add_clip("long", None, 0, 30, 0.5, None, [{"source_id": sid, "start": 50, "duration": 30}])
    assert sorted(db.used_ranges()[sid]) == [(10, 25), (50, 80)]
    db.update_clip(c, status="rejected")
    assert db.used_ranges()[sid] == [(50, 80)]


# ---------------------------------------------------------------- render (integration)
def test_render_short_end_to_end(tmp_path):
    from moto_shorts.edit.render import RenderSettings, render_short, render_social
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=4",
                    "-f", "lavfi", "-i", "sine=f=120:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(src)], check=True)
    s = RenderSettings(encoder="libx264", short_size=(180, 320), social_size=(90, 160), bitrate_master="2M",
                       bitrate_social="1M")
    m = render_short(src, 0.5, 2.5, tmp_path / "m.mp4", s, Grade())
    o = render_social(m, tmp_path / "s.mp4", s)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(o)], capture_output=True, text=True).stdout
    assert out.strip() == "90,160"


def test_instagram_rotated_is_vertical_full_frame(tmp_path):
    from moto_shorts.edit.render import RenderSettings, render_instagram_rotated
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=3",
                    "-f", "lavfi", "-i", "sine=f=120:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(src)], check=True)
    s = RenderSettings(encoder="libx264", social_size=(180, 320), bitrate_social="1M")
    o = render_instagram_rotated(src, 0.2, 2.0, tmp_path / "ig.mp4", s, Grade())
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(o)], capture_output=True, text=True).stdout
    assert out.strip() == "180,320"


def test_vertical_graph_labels():
    g = vertical_graph(2160, 3840, 12, Grade())
    assert "[vout]" in g and "overlay" in g and "gblur" in g and "crop=2160:1212" in g
