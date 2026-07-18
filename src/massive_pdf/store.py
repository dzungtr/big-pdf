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
# Slice 3 (issue #4) helpers: clause graph (clauses + clause_references)
# ---------------------------------------------------------------------------

# NOTE: The slice-3 brief (issue #4) calls for tables `clauses` and
# `clause_references` instead of the slice-1 `sections`/`edges` placeholders.
# We add them additively (SCHEMA_V3) so an existing slice-2 DB migrates
# cleanly. Old tables stay for back-compat until slice-5/6 explicitly retire
# them.

SCHEMA_V3 = """
-- Native clause hierarchy: Điều -> Khoản -> Điểm.
-- `kind` is one of 'dieu' | 'khoan' | 'diem'.
-- `number` is the local label ('1', '2', 'a', 'b', ...).
-- `citation` is the canonical human-readable citation
--   ('Điều 1', 'Khoản 2 Điều 3', 'Điểm a Khoản 2 Điều 3').
-- `parent_id` links a Khoản to its Điều, a Điểm to its Khoản.
CREATE TABLE IF NOT EXISTS clauses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    parent_id INTEGER REFERENCES clauses(id),
    kind TEXT NOT NULL CHECK(kind IN ('dieu','khoan','diem')),
    number TEXT NOT NULL,
    citation TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    body TEXT NOT NULL,
    ord INTEGER NOT NULL,
    UNIQUE(document_id, citation)
);
CREATE INDEX IF NOT EXISTS idx_clauses_document ON clauses(document_id);
CREATE INDEX IF NOT EXISTS idx_clauses_parent ON clauses(parent_id);
CREATE INDEX IF NOT EXISTS idx_clauses_doc_ord ON clauses(document_id, ord);

-- Cross-references between clauses (internal) or to external instruments
-- (dangling). Internal: dst_clause_id is set. External: dst_clause_id is
-- NULL and target_citation/target_document_id carry the external pointer.
CREATE TABLE IF NOT EXISTS clause_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_clause_id INTEGER NOT NULL REFERENCES clauses(id),
    dst_clause_id INTEGER REFERENCES clauses(id),
    kind TEXT NOT NULL CHECK(kind IN ('internal','external')),
    raw_text TEXT NOT NULL,
    target_citation TEXT NOT NULL,
    target_document_id INTEGER REFERENCES documents(id),
    UNIQUE(src_clause_id, raw_text, target_citation)
);
CREATE INDEX IF NOT EXISTS idx_clause_refs_src ON clause_references(src_clause_id);
CREATE INDEX IF NOT EXISTS idx_clause_refs_dst ON clause_references(dst_clause_id);
CREATE INDEX IF NOT EXISTS idx_clause_refs_target_doc ON clause_references(target_document_id);
"""

SCHEMA = SCHEMA_V1 + SCHEMA_V2 + SCHEMA_V3
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


# ---------------------------------------------------------------------------
# Slice 2 helpers: document lookup
# ---------------------------------------------------------------------------

def get_document(conn, document_id: int):
    row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    return dict(row) if row else None


def get_document_by_path(conn, source_path: str):
    row = conn.execute("SELECT * FROM documents WHERE source_path=?", (source_path,)).fetchone()
    return dict(row) if row else None

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Slice 3 helpers: clause CRUD + reference CRUD
# ---------------------------------------------------------------------------

def upsert_clause(conn, *, document_id: int, parent_id: int | None,
                  kind: str, number: str, citation: str,
                  page_start: int, page_end: int, body: str,
                  ord: int) -> int:
    """Insert or update a clause keyed on (document_id, citation).

    Returns the clause id. Updating an existing row preserves its id so
    references remain stable across re-runs.
    """
    if kind not in ("dieu", "khoan", "diem"):
        raise ValueError(f"clause kind must be dieu|khoan|diem, got {kind!r}")
    conn.execute(
        """
        INSERT INTO clauses(document_id, parent_id, kind, number, citation,
                            page_start, page_end, body, ord)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, citation) DO UPDATE SET
            parent_id=excluded.parent_id,
            kind=excluded.kind,
            number=excluded.number,
            page_start=excluded.page_start,
            page_end=excluded.page_end,
            body=excluded.body,
            ord=excluded.ord
        """,
        (document_id, parent_id, kind, number, citation,
         page_start, page_end, body, ord),
    )
    row = conn.execute(
        "SELECT id FROM clauses WHERE document_id=? AND citation=?",
        (document_id, citation),
    ).fetchone()
    return row[0]


def get_clause(conn, clause_id: int):
    row = conn.execute("SELECT * FROM clauses WHERE id=?", (clause_id,)).fetchone()
    return dict(row) if row else None


def get_clause_by_citation(conn, document_id: int, citation: str):
    row = conn.execute(
        "SELECT * FROM clauses WHERE document_id=? AND citation=?",
        (document_id, citation),
    ).fetchone()
    return dict(row) if row else None


def list_clauses(conn, document_id: int):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM clauses WHERE document_id=? ORDER BY ord",
        (document_id,),
    ).fetchall()]


def list_clause_children(conn, clause_id: int):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM clauses WHERE parent_id=? ORDER BY ord",
        (clause_id,),
    ).fetchall()]


def delete_clauses_for_document(conn, document_id: int) -> int:
    """Hard-delete a document's clauses + their cross-references.

    Used by the resumable `structure` stage when the input changes and
    a clean rebuild is required. Returns total clauses deleted.
    """
    cur = conn.execute("DELETE FROM clause_references WHERE src_clause_id IN "
                       "(SELECT id FROM clauses WHERE document_id=?)",
                       (document_id,))
    refs_deleted = cur.rowcount
    cur = conn.execute("DELETE FROM clauses WHERE document_id=?", (document_id,))
    clauses_deleted = cur.rowcount
    return clauses_deleted


def add_clause_reference(conn, *, src_clause_id: int,
                         dst_clause_id: int | None,
                         kind: str, raw_text: str, target_citation: str,
                         target_document_id: int | None) -> int | None:
    """Insert a cross-reference. Deduplicates on (src, raw_text, target_citation).

    Returns the new row id, or None if a duplicate was suppressed.
    """
    if kind not in ("internal", "external"):
        raise ValueError(f"ref kind must be internal|external, got {kind!r}")
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO clause_references(
            src_clause_id, dst_clause_id, kind, raw_text,
            target_citation, target_document_id
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (src_clause_id, dst_clause_id, kind, raw_text,
         target_citation, target_document_id),
    )
    # SQLite keeps `lastrowid` at the previous successful insert when
    # INSERT OR IGNORE skips a row, so `rowcount` is the reliable
    # dedup signal.
    if cur.rowcount == 0:
        return None
    return cur.lastrowid


def list_clause_references(conn, clause_id: int, direction: str = "out"):
    """List references where clause_id is the source ('out') or destination ('in')."""
    if direction == "out":
        rows = conn.execute(
            "SELECT * FROM clause_references WHERE src_clause_id=? "
            "ORDER BY id",
            (clause_id,),
        ).fetchall()
    elif direction == "in":
        rows = conn.execute(
            "SELECT * FROM clause_references WHERE dst_clause_id=? "
            "ORDER BY id",
            (clause_id,),
        ).fetchall()
    else:
        raise ValueError(f"direction must be in|out, got {direction!r}")
    return [dict(r) for r in rows]


def count_clauses_by_kind(conn, document_id: int) -> dict[str, int]:
    rows = conn.execute(
        "SELECT kind, COUNT(*) FROM clauses WHERE document_id=? GROUP BY kind",
        (document_id,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def dangling_references(conn, document_id: int) -> list[dict]:
    """External/dangling refs (dst_clause_id IS NULL) for a document."""
    rows = conn.execute(
        """
        SELECT cr.* FROM clause_references cr
        JOIN clauses c ON c.id = cr.src_clause_id
        WHERE c.document_id=? AND cr.dst_clause_id IS NULL
        ORDER BY cr.id
        """,
        (document_id,),
    ).fetchall()
    return [dict(r) for r in rows]
