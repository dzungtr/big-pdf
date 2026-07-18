"""CLI for the massive-pdf-agent: register documents and run ingest stages.

Subcommands (slice 2 — issue #3):
  register   Insert a new document record (path + title) into the store.
  list       List registered documents.
  pages      Run the `pages` ingest stage (PDF -> PNG images).
  ocr        Run the `ocr` ingest stage (page images -> text+layout JSON).

With no subcommand, falls back to the slice-1 smoke behaviour so the
existing `test_smoke.py` keeps passing.
"""
import argparse
import sys
from pathlib import Path

from .store import (
    connect,
    get_document,
    get_document_by_path,
    init_db,
    insert_document,
    list_documents,
)
from .ingest.pages import DEFAULT_DPI, run_pages_stage
from .ingest.ocr import StubBackend, run_ocr_stage


def _ensure_document(db: str, source_path: str, title: str | None) -> int:
    """Return doc id for `source_path`, inserting if absent."""
    init_db(db)
    with connect(db) as conn:
        existing = get_document_by_path(conn, source_path)
        if existing:
            return existing["id"]
        did = insert_document(conn, source_path, title=title)
        conn.commit()
    return did


def cmd_register(args) -> int:
    did = _ensure_document(args.db, args.path, args.title)
    print(f"registered doc_id={did} path={args.path} title={args.title}")
    return 0


def cmd_list(args) -> int:
    init_db(args.db)
    with connect(args.db) as conn:
        docs = list_documents(conn)
    for d in docs:
        print(f"id={d['id']} title={d['title']} path={d['source_path']}")
    print(f"total docs={len(docs)}")
    return 0


def cmd_pages(args) -> int:
    did = _ensure_document(args.db, args.pdf, args.title)
    n = run_pages_stage(args.db, did, args.pdf, args.out, dpi=args.dpi)
    print(f"pages stage: doc_id={did} rendered={n} dpi={args.dpi} out={args.out}")
    return 0


def cmd_ocr(args) -> int:
    init_db(args.db)
    with connect(args.db) as conn:
        doc = get_document(conn, args.doc_id)
        if doc is None:
            print(f"ocr stage: no document with id={args.doc_id}", file=sys.stderr)
            return 2
    n = run_ocr_stage(args.db, args.doc_id, args.out, backend=StubBackend())
    print(f"ocr stage: doc_id={args.doc_id} pages={n} out={args.out}")
    return 0


def cmd_smoke(args) -> int:
    """Backwards-compatible smoke: init DB, insert a demo doc, list docs."""
    init_db(args.db)
    with connect(args.db) as conn:
        did = insert_document(
            conn, "Thong-tu-89-BTC.pdf", title="Thông tư 89/2015/TT-BTC"
        )
        conn.commit()
        docs = list_documents(conn)
    print(f"inserted doc_id={did}; total docs={len(docs)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="massive-pdf")
    parser.add_argument("--db", default=".massive_pdf.sqlite")
    sub = parser.add_subparsers(dest="command")

    p_reg = sub.add_parser("register", help="Register a document in the store")
    p_reg.add_argument("path", help="Path to the source PDF")
    p_reg.add_argument("--title", default=None, help="Human-readable title")

    sub.add_parser("list", help="List registered documents")

    p_pages = sub.add_parser(
        "pages", help="Render PDF pages to images (slice 2 stage 1)"
    )
    p_pages.add_argument("pdf", help="Path to source PDF")
    p_pages.add_argument(
        "--out", default=".massive_pdf/pages", help="Output directory for PNGs"
    )
    p_pages.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI, help="Render DPI (default 300)"
    )
    p_pages.add_argument("--title", default=None, help="Document title")

    p_ocr = sub.add_parser(
        "ocr", help="Run OCR over a document's page images (slice 2 stage 2)"
    )
    p_ocr.add_argument("doc_id", type=int, help="Document id (see `massive-pdf list`)")
    p_ocr.add_argument(
        "--out", default=".massive_pdf", help="Output root for OCR artifacts"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        return cmd_smoke(args)

    return {
        "register": cmd_register,
        "list": cmd_list,
        "pages": cmd_pages,
        "ocr": cmd_ocr,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
