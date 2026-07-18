"""Tests for the new rule_cards store helpers (slice 5, issue #5)."""
import pytest

from massive_pdf.store import (
    update_rule_card_embedding,
    connect,
    count_rule_cards_by_document,
    init_db,
    insert_document,
    insert_rule_card,
    list_rule_cards_for_clause,
    list_rule_cards_for_document,
)


@pytest.fixture
def doc_with_clauses(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    with connect(db) as c:
        did = insert_document(c, "doc.pdf", title="rule-card helpers test")
        d1 = _insert_clause(c, did, parent=None, kind="dieu", number="1",
                            citation="Điều 1", body="Điều 1. ...", ord=1)
        d2 = _insert_clause(c, did, parent=None, kind="dieu", number="2",
                            citation="Điều 2", body="Điều 2. ...", ord=2)
        c.commit()
    return db, did, d1, d2


def _insert_clause(conn, doc_id, *, parent, kind, number, citation, body, ord):
    cur = conn.execute(
        """
        INSERT INTO clauses(document_id, parent_id, kind, number, citation,
                            page_start, page_end, body, ord)
        VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?)
        """,
        (doc_id, parent, kind, number, citation, body, ord),
    )
    return cur.lastrowid


def test_rule_cards_table_has_clause_id_column(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    with connect(db) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(rule_cards)").fetchall()}
    assert "clause_id" in cols


def test_init_db_migration_is_idempotent(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    init_db(db)
    init_db(db)
    # Still no error; column + index present.
    with connect(db) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(rule_cards)").fetchall()}
        idx = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
    assert "clause_id" in cols
    assert "idx_rule_cards_clause" in idx


def test_insert_and_list_rule_cards(doc_with_clauses):
    db, did, d1, d2 = doc_with_clauses
    with connect(db) as c:
        rid_a = insert_rule_card(c, clause_id=d1, statement="Điều 1: gloss A")
        rid_b = insert_rule_card(c, clause_id=d1, statement="Điều 1: gloss B")
        rid_c = insert_rule_card(c, clause_id=d2, statement="Điều 2: gloss C")
        c.commit()

        d1_cards = list_rule_cards_for_clause(c, d1)
        d2_cards = list_rule_cards_for_clause(c, d2)
        assert {r["id"] for r in d1_cards} == {rid_a, rid_b}
        assert {r["id"] for r in d2_cards} == {rid_c}

        all_cards = list_rule_cards_for_document(c, did)
        assert {r["id"] for r in all_cards} == {rid_a, rid_b, rid_c}


def test_update_rule_card_embedding_writes_blob(doc_with_clauses):
    db, did, d1, _ = doc_with_clauses
    with connect(db) as c:
        rid = insert_rule_card(c, clause_id=d1, statement="Điều 1: gloss")
        c.commit()
    blob = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    with connect(db) as c:
        update_rule_card_embedding(c, rid, blob)
        c.commit()
    with connect(db) as c:
        row = c.execute("SELECT embedding FROM rule_cards WHERE id=?", (rid,)).fetchone()
    assert row[0] == blob


def test_count_rule_cards_by_document(doc_with_clauses):
    db, did, d1, d2 = doc_with_clauses
    with connect(db) as c:
        insert_rule_card(c, clause_id=d1, statement="A")
        insert_rule_card(c, clause_id=d1, statement="B")
        insert_rule_card(c, clause_id=d2, statement="C")
        c.commit()
        n = count_rule_cards_by_document(c, did)
    assert n == 3
