"""SQLite store for documents, sections, edges, rule_cards. Plain SQL schema + connection helper."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL UNIQUE,
    title TEXT,
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    ordinal INTEGER NOT NULL,
    heading TEXT,
    body TEXT,
    UNIQUE(document_id, ordinal)
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_section_id INTEGER NOT NULL REFERENCES sections(id),
    dst_section_id INTEGER NOT NULL REFERENCES sections(id),
    kind TEXT NOT NULL,
    UNIQUE(src_section_id, dst_section_id, kind)
);
CREATE TABLE IF NOT EXISTS rule_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id INTEGER NOT NULL REFERENCES sections(id),
    statement TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_section_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_section_id);
"""

@contextmanager
def connect(db_path: str | Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()

def insert_document(conn, source_path: str, title: str | None = None) -> int:
    cur = conn.execute("INSERT INTO documents(source_path, title) VALUES (?, ?)", (source_path, title))
    return cur.lastrowid

def list_documents(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM documents ORDER BY id").fetchall()]
