"""SQLite store for documents, pages, sections, edges, rule_cards, and ingest checkpoints.

Slice 1 introduced documents / sections / edges / rule_cards (see `SCHEMA_V1`).
Slice 2 (`#3` — OCR stage) adds `pages` (rendered page-image registry) and
`ingest_checkpoints` (per-stage, per-page resumability markers). The two
schemas are applied in order so an existing slice-1 DB migrates cleanly.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA_V1 = """
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

# Slice 2: per-document page-image registry and per-stage resumability.
SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    page_ordinal INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    dpi INTEGER NOT NULL,
    UNIQUE(document_id, page_ordinal)
);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);

-- One row per (document, stage, page_ordinal). page_ordinal=0 means
-- "document-level marker" (e.g. ocr stage finished all pages).
CREATE TABLE IF NOT EXISTS ingest_checkpoints (
    document_id INTEGER NOT NULL REFERENCES documents(id),
    stage TEXT NOT NULL,
    page_ordinal INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('done','failed')),
    artifact_path TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, stage, page_ordinal)
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_stage ON ingest_checkpoints(stage, status);
"""

SCHEMA = SCHEMA_V1 + SCHEMA_V2


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
    cur = conn.execute(
        "INSERT INTO documents(source_path, title) VALUES (?, ?)",
        (source_path, title),
    )
    return cur.lastrowid


def list_documents(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM documents ORDER BY id").fetchall()]


# ---------------------------------------------------------------------------
# Slice 2 helpers: pages registry + ingest checkpoints
# ---------------------------------------------------------------------------

def upsert_page(conn, document_id: int, page_ordinal: int, image_path: str,
                width: int, height: int, dpi: int) -> int:
    """Insert or update a page-image registry row; returns page id."""
    conn.execute(
        """
        INSERT INTO pages(document_id, page_ordinal, image_path, width, height, dpi)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, page_ordinal) DO UPDATE SET
            image_path=excluded.image_path,
            width=excluded.width,
            height=excluded.height,
            dpi=excluded.dpi
        """,
        (document_id, page_ordinal, image_path, width, height, dpi),
    )
    return conn.execute(
        "SELECT id FROM pages WHERE document_id=? AND page_ordinal=?",
        (document_id, page_ordinal),
    ).fetchone()[0]


def get_page(conn, document_id: int, page_ordinal: int):
    row = conn.execute(
        "SELECT * FROM pages WHERE document_id=? AND page_ordinal=?",
        (document_id, page_ordinal),
    ).fetchone()
    return dict(row) if row else None


def list_pages(conn, document_id: int):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM pages WHERE document_id=? ORDER BY page_ordinal",
        (document_id,),
    ).fetchall()]


def mark_checkpoint(conn, document_id: int, stage: str, page_ordinal: int,
                   status: str, artifact_path: str | None = None) -> None:
    """Upsert a per-(doc, stage, page) checkpoint. status in {'done','failed'}."""
    if status not in ("done", "failed"):
        raise ValueError(f"checkpoint status must be done|failed, got {status!r}")
    conn.execute(
        """
        INSERT INTO ingest_checkpoints(document_id, stage, page_ordinal, status, artifact_path)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(document_id, stage, page_ordinal) DO UPDATE SET
            status=excluded.status,
            artifact_path=excluded.artifact_path,
            updated_at=CURRENT_TIMESTAMP
        """,
        (document_id, stage, page_ordinal, status, artifact_path),
    )


def get_checkpoint(conn, document_id: int, stage: str, page_ordinal: int):
    row = conn.execute(
        "SELECT * FROM ingest_checkpoints WHERE document_id=? AND stage=? AND page_ordinal=?",
        (document_id, stage, page_ordinal),
    ).fetchone()
    return dict(row) if row else None


def completed_page_ordinals(conn, document_id: int, stage: str) -> set[int]:
    """Return the set of page_ordinals marked 'done' for a given doc+stage.

    Used by resumable stages to skip work that's already finished.
    """
    rows = conn.execute(
        """
        SELECT page_ordinal FROM ingest_checkpoints
        WHERE document_id=? AND stage=? AND status='done' AND page_ordinal > 0
        """,
        (document_id, stage),
    ).fetchall()
    return {r[0] for r in rows}
