import os, tempfile
from massive_pdf.store import init_db, insert_document, list_documents, connect

def test_schema_roundtrip(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    with connect(db) as conn:
        did = insert_document(conn, "doc.pdf", title="hello")
        conn.commit()
        rows = list_documents(conn)
    assert rows == [{"id": 1, "source_path": "doc.pdf", "title": "hello", "ingested_at": rows[0]["ingested_at"]}]

def test_init_db_idempotent(tmp_path):
    db = tmp_path / "t.sqlite"
    init_db(db)
    init_db(db)  # should not raise

def test_cli_smoke(tmp_path, monkeypatch, capsys):
    db = tmp_path / "t.sqlite"
    from massive_pdf.__main__ import main
    rc = main(["--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "inserted doc_id=" in out
    assert "total docs=" in out
