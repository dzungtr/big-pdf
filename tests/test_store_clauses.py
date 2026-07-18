"""Tests for slice-3 store additions: clauses + clause_references."""
import pytest
from massive_pdf.store import (
    add_clause_reference,
    connect,
    count_clauses_by_kind,
    dangling_references,
    delete_clauses_for_document,
    get_clause_by_citation,
    init_db,
    insert_document,
    list_clause_children,
    list_clause_references,
    list_clauses,
    upsert_clause,
)


@pytest.fixture
def doc_with_clauses(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    with connect(db) as c:
        doc_id = insert_document(c, "doc.pdf", title="hello")
        d1 = upsert_clause(c, document_id=doc_id, parent_id=None,
                           kind="dieu", number="1", citation="Điều 1",
                           page_start=1, page_end=2, body="Điều 1.", ord=1)
        k1 = upsert_clause(c, document_id=doc_id, parent_id=d1,
                           kind="khoan", number="1", citation="Khoản 1 Điều 1",
                           page_start=1, page_end=1, body="Khoản 1.", ord=2)
        d2 = upsert_clause(c, document_id=doc_id, parent_id=None,
                           kind="dieu", number="2", citation="Điều 2",
                           page_start=2, page_end=3, body="Điều 2.", ord=3)
        c.commit()
    return db, doc_id, {"Điều 1": d1, "Khoản 1 Điều 1": k1, "Điều 2": d2}


def test_clauses_table_exists(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    with connect(db) as c:
        names = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "clauses" in names
    assert "clause_references" in names


def test_upsert_clause_inserts_and_dedupes(doc_with_clauses):
    db, doc_id, ids = doc_with_clauses
    with connect(db) as c:
        # Re-upsert Điều 1 with new body — id should stay, body should update.
        same_id = upsert_clause(c, document_id=doc_id, parent_id=None,
                                kind="dieu", number="1", citation="Điều 1",
                                page_start=1, page_end=4, body="UPDATED",
                                ord=1)
        assert same_id == ids["Điều 1"]
        row = get_clause_by_citation(c, doc_id, "Điều 1")
        assert row["body"] == "UPDATED"
        assert row["page_end"] == 4


def test_kind_constraint_enforced(doc_with_clauses):
    db, doc_id, _ = doc_with_clauses
    with connect(db) as c, pytest.raises(Exception):
        upsert_clause(c, document_id=doc_id, parent_id=None,
                      kind="bogus", number="1", citation="bogus",
                      page_start=1, page_end=1, body="x", ord=99)


def test_list_clauses_ordered(doc_with_clauses):
    db, doc_id, _ = doc_with_clauses
    with connect(db) as c:
        clauses = list_clauses(c, doc_id)
        assert [r["citation"] for r in clauses] == [
            "Điều 1", "Khoản 1 Điều 1", "Điều 2",
        ]


def test_list_clause_children(doc_with_clauses):
    db, doc_id, ids = doc_with_clauses
    with connect(db) as c:
        kids = list_clause_children(c, ids["Điều 1"])
        assert [r["citation"] for r in kids] == ["Khoản 1 Điều 1"]


def test_count_clauses_by_kind(doc_with_clauses):
    db, doc_id, _ = doc_with_clauses
    with connect(db) as c:
        counts = count_clauses_by_kind(c, doc_id)
        assert counts == {"dieu": 2, "khoan": 1}


def test_add_clause_reference_internal_and_dedup(doc_with_clauses):
    db, doc_id, ids = doc_with_clauses
    with connect(db) as c:
        rid1 = add_clause_reference(c, src_clause_id=ids["Điều 2"],
                                    dst_clause_id=ids["Điều 1"],
                                    kind="internal", raw_text="Điều 1",
                                    target_citation="Điều 1",
                                    target_document_id=None)
        assert rid1 is not None
        # Dedup: same (src, raw_text, target_citation) → None.
        rid2 = add_clause_reference(c, src_clause_id=ids["Điều 2"],
                                    dst_clause_id=ids["Điều 1"],
                                    kind="internal", raw_text="Điều 1",
                                    target_citation="Điều 1",
                                    target_document_id=None)
        assert rid2 is None
        c.commit()
        rows = list_clause_references(c, ids["Điều 2"], direction="out")
        assert len(rows) == 1


def test_add_clause_reference_external_dangling(doc_with_clauses):
    db, doc_id, ids = doc_with_clauses
    with connect(db) as c:
        add_clause_reference(c, src_clause_id=ids["Điều 1"],
                             dst_clause_id=None, kind="external",
                             raw_text="Thông tư số 39/2014/TT-BTC",
                             target_citation="Thông tư số 39/2014/TT-BTC",
                             target_document_id=None)
        c.commit()
        dangling = dangling_references(c, doc_id)
        assert len(dangling) == 1
        assert dangling[0]["kind"] == "external"


def test_references_in_direction(doc_with_clauses):
    db, doc_id, ids = doc_with_clauses
    with connect(db) as c:
        add_clause_reference(c, src_clause_id=ids["Điều 2"],
                             dst_clause_id=ids["Điều 1"],
                             kind="internal", raw_text="Điều 1",
                             target_citation="Điều 1", target_document_id=None)
        c.commit()
        out_rows = list_clause_references(c, ids["Điều 2"], direction="out")
        in_rows = list_clause_references(c, ids["Điều 1"], direction="in")
        assert len(out_rows) == 1
        assert len(in_rows) == 1
        # Same edge row must be reachable from both src (out) and dst (in).
        assert out_rows[0] == in_rows[0]


def test_delete_clauses_for_document(doc_with_clauses):
    db, doc_id, ids = doc_with_clauses
    with connect(db) as c:
        add_clause_reference(c, src_clause_id=ids["Điều 2"],
                             dst_clause_id=ids["Điều 1"],
                             kind="internal", raw_text="Điều 1",
                             target_citation="Điều 1", target_document_id=None)
        c.commit()
    with connect(db) as c:
        n = delete_clauses_for_document(c, doc_id)
        assert n == 3
        c.commit()
        assert list_clauses(c, doc_id) == []
        assert dangling_references(c, doc_id) == []
