"""Tests for slice-2 store additions: pages registry + ingest checkpoints."""
import os
import pytest
from massive_pdf.store import (
    connect,
    init_db,
    insert_document,
    upsert_page,
    get_page,
    list_pages,
    mark_checkpoint,
    get_checkpoint,
    completed_page_ordinals,
)


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    with connect(db) as c:
        doc_id = insert_document(c, "doc.pdf", title="hello")
        c.commit()
        yield c, doc_id


def test_pages_and_checkpoints_tables_exist(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    with connect(db) as c:
        names = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "pages" in names
    assert "ingest_checkpoints" in names


def test_upsert_page_inserts_and_updates(conn):
    c, doc_id = conn
    pid1 = upsert_page(c, doc_id, 1, "/tmp/p1.png", 1200, 1700, 300)
    pid2 = upsert_page(c, doc_id, 2, "/tmp/p2.png", 1200, 1700, 300)
    assert pid1 != pid2
    # Upsert same key → id stays the same, image_path updates.
    pid1b = upsert_page(c, doc_id, 1, "/tmp/p1_v2.png", 1200, 1700, 300)
    assert pid1b == pid1
    row = get_page(c, doc_id, 1)
    assert row["image_path"] == "/tmp/p1_v2.png"


def test_list_pages_ordered(conn):
    c, doc_id = conn
    for o in (3, 1, 2):
        upsert_page(c, doc_id, o, f"/tmp/p{o}.png", 100, 200, 300)
    rows = list_pages(c, doc_id)
    assert [r["page_ordinal"] for r in rows] == [1, 2, 3]


def test_mark_and_get_checkpoint(conn):
    c, doc_id = conn
    mark_checkpoint(c, doc_id, "pages", 1, "done", artifact_path="/tmp/p1.png")
    mark_checkpoint(c, doc_id, "ocr", 1, "failed", artifact_path=None)
    p = get_checkpoint(c, doc_id, "pages", 1)
    assert p["status"] == "done" and p["artifact_path"] == "/tmp/p1.png"
    o = get_checkpoint(c, doc_id, "ocr", 1)
    assert o["status"] == "failed" and o["artifact_path"] is None


def test_checkpoint_status_is_validated(conn):
    c, doc_id = conn
    with pytest.raises(ValueError):
        mark_checkpoint(c, doc_id, "pages", 1, "bogus")


def test_completed_page_ordinals_skips_zero(conn):
    c, doc_id = conn
    for o in (1, 2, 3, 5):
        mark_checkpoint(c, doc_id, "ocr", o, "done")
    mark_checkpoint(c, doc_id, "ocr", 0, "done")  # doc-level marker
    mark_checkpoint(c, doc_id, "ocr", 4, "failed")  # should be excluded
    assert completed_page_ordinals(c, doc_id, "ocr") == {1, 2, 3, 5}


def test_init_db_idempotent_for_v2(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    init_db(db)  # should not raise on re-run with V2 tables
    with connect(db) as c:
        cnt = c.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        assert cnt == 0
