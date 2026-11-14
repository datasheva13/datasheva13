"""SQLite state: source files, generated clips, approvals, what's already been used."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY,
  gopro_id TEXT UNIQUE,
  filename TEXT,
  path TEXT,
  captured_at TEXT,
  duration REAL,
  status TEXT DEFAULT 'new',          -- new | analyzed | failed | purged
  analysis_path TEXT,
  error TEXT,
  added_at TEXT
);
CREATE TABLE IF NOT EXISTS clips (
  id INTEGER PRIMARY KEY,
  kind TEXT,                          -- short | long
  source_id INTEGER,
  start REAL,
  duration REAL,
  segments TEXT,                      -- json for long videos
  score REAL,
  embedding BLOB,
  status TEXT DEFAULT 'pending',      -- pending | rendered | approved | posted | rejected | failed
  master_path TEXT,
  social_path TEXT,
  instagram_path TEXT,
  thumb_options TEXT,                 -- json list of frame paths
  thumb_path TEXT,
  captions TEXT,                      -- json from captions.generate
  hook TEXT,
  title TEXT,
  description TEXT,
  hashtags TEXT,                      -- json list of 3
  thumb_word TEXT,
  grade_strength REAL DEFAULT 1.0,
  music TEXT,
  youtube_id TEXT,
  error TEXT,
  created_at TEXT,
  posted_at TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(clips)")}
        if "instagram_path" not in cols:                    # upgrade older databases
            self.conn.execute("ALTER TABLE clips ADD COLUMN instagram_path TEXT")
        self.conn.commit()

    # sources -------------------------------------------------------------
    def add_source(self, gopro_id: str, filename: str, path: str, captured_at: str | None, duration: float) -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO sources (gopro_id, filename, path, captured_at, duration, added_at) VALUES (?,?,?,?,?,?)",
            (gopro_id, filename, path, captured_at, duration, now()))
        self.conn.commit()
        if cur.lastrowid and cur.rowcount:
            return cur.lastrowid
        return self.conn.execute("SELECT id FROM sources WHERE gopro_id=?", (gopro_id,)).fetchone()["id"]

    def has_source(self, gopro_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sources WHERE gopro_id=?", (gopro_id,)).fetchone() is not None

    def sources(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.conn.execute("SELECT * FROM sources WHERE status=? ORDER BY captured_at", (status,)).fetchall()
        return self.conn.execute("SELECT * FROM sources ORDER BY captured_at").fetchall()

    def source(self, sid: int) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM sources WHERE id=?", (sid,)).fetchone()

    def update_source(self, sid: int, **fields) -> None:
        self._update("sources", sid, fields)

    # clips ---------------------------------------------------------------
    def add_clip(self, kind: str, source_id: int | None, start: float, duration: float, score: float,
                 embedding: np.ndarray | None, segments: list | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO clips (kind, source_id, start, duration, score, embedding, segments, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (kind, source_id, start, duration, score,
             embedding.astype(np.float32).tobytes() if embedding is not None else None,
             json.dumps(segments) if segments else None, now()))
        self.conn.commit()
        return cur.lastrowid

    def clip(self, cid: int) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM clips WHERE id=?", (cid,)).fetchone()

    def clips(self, statuses: tuple[str, ...] = ("rendered",), kind: str | None = None) -> list[sqlite3.Row]:
        q = f"SELECT * FROM clips WHERE status IN ({','.join('?' * len(statuses))})"
        args: list = list(statuses)
        if kind:
            q += " AND kind=?"
            args.append(kind)
        return self.conn.execute(q + " ORDER BY created_at DESC, id", args).fetchall()

    def update_clip(self, cid: int, **fields) -> None:
        for k in ("captions", "hashtags", "thumb_options", "segments"):
            if k in fields and not isinstance(fields[k], (str, type(None))):
                fields[k] = json.dumps(fields[k])
        self._update("clips", cid, fields)

    def used_ranges(self) -> dict[int, list[tuple[float, float]]]:
        """Time ranges already turned into shorts or long videos (rejected ones are free again)."""
        out: dict[int, list[tuple[float, float]]] = {}
        for r in self.conn.execute("SELECT * FROM clips WHERE status NOT IN ('rejected','failed')"):
            if r["kind"] == "short" and r["source_id"] is not None:
                out.setdefault(r["source_id"], []).append((r["start"], r["start"] + r["duration"]))
            elif r["segments"]:
                for seg in json.loads(r["segments"]):
                    out.setdefault(seg["source_id"], []).append((seg["start"], seg["start"] + seg["duration"]))
        return out

    def recent_embeddings(self, days: int = 10, limit: int = 30) -> list[np.ndarray]:
        rows = self.conn.execute(
            "SELECT embedding FROM clips WHERE kind='short' AND embedding IS NOT NULL AND status NOT IN ('rejected','failed') "
            "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]

    def _update(self, table: str, rid: int, fields: dict) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE {table} SET {cols} WHERE id=?", (*fields.values(), rid))
        self.conn.commit()
