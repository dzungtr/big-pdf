"""Smoke CLI: create an empty DB at the given path, insert a doc, list docs, exit."""
import argparse, sys
from pathlib import Path
from .store import init_db, insert_document, list_documents, connect

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="massive-pdf")
    parser.add_argument("--db", default=".massive_pdf.sqlite")
    args = parser.parse_args(argv)
    init_db(args.db)
    with connect(args.db) as conn:
        did = insert_document(conn, "Thong-tu-89-BTC.pdf", title="Thông tư 89/2015/TT-BTC")
        conn.commit()
        docs = list_documents(conn)
    print(f"inserted doc_id={did}; total docs={len(docs)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
