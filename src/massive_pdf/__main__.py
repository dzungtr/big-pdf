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
from .ingest.ocr import StubBackend, default_backend, run_ocr_stage
from .ingest.vlm import UnlimitedOcrBackend
from .structure import check_invariants, run_structure_stage
from .retrieval import (
    HashBagEncoder,
    get_default_encoder,
    run_cards_stage,
    run_embed_stage,
)


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
    if args.backend == "vlm":
        backend = UnlimitedOcrBackend()
    elif args.backend == "stub":
        backend = StubBackend()
    else:
        backend = default_backend()
    n = run_ocr_stage(args.db, args.doc_id, args.out, backend=backend)
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


# ---------------------------------------------------------------------------
# Slice 3 (issue #4): structure + invariants subcommands
# ---------------------------------------------------------------------------

def cmd_structure(args) -> int:
    init_db(args.db)
    with connect(args.db) as conn:
        doc = get_document(conn, args.doc_id)
        if doc is None:
            print(f"structure stage: no document with id={args.doc_id}",
                  file=sys.stderr)
            return 2
    result = run_structure_stage(args.db, args.doc_id, args.out,
                                 rebuild=args.rebuild)
    print(
        f"structure stage: doc_id={args.doc_id} clauses={result['clauses']} "
        f"refs_internal={result['refs_internal']} "
        f"refs_external={result['refs_external']} "
        f"pages={result['pages']} out={args.out}"
    )
    return 0


def cmd_invariants(args) -> int:
    init_db(args.db)
    with connect(args.db) as conn:
        doc = get_document(conn, args.doc_id)
        if doc is None:
            print(f"invariants: no document with id={args.doc_id}",
                  file=sys.stderr)
            return 2
    report = check_invariants(args.db, args.doc_id)
    print(str(report))
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# Slice 5 (issue #5): rule-cards subcommands
# ---------------------------------------------------------------------------

def cmd_cards(args) -> int:
    init_db(args.db)
    with connect(args.db) as conn:
        doc = get_document(conn, args.doc_id)
        if doc is None:
            print(f"cards stage: no document with id={args.doc_id}",
                  file=sys.stderr)
            return 2
    result = run_cards_stage(args.db, args.doc_id, args.out,
                             rebuild=args.rebuild)
    status = "done" if result["ok"] else "with-errors"
    print(
        f"cards stage: doc_id={args.doc_id} "
        f"clauses_scanned={result['clauses_scanned']} "
        f"cards_inserted={result['cards_inserted']} "
        f"skipped={result['skipped']} "
        f"failed={len(result['failed'])} "
        f"status={status} out={args.out}"
    )
    return 0 if result["ok"] else 1


def cmd_embed(args) -> int:
    init_db(args.db)
    with connect(args.db) as conn:
        doc = get_document(conn, args.doc_id)
        if doc is None:
            print(f"embed stage: no document with id={args.doc_id}",
                  file=sys.stderr)
            return 2
    encoder = get_default_encoder(dim=args.dim)
    result = run_embed_stage(args.db, args.doc_id, encoder,
                             rebuild=args.rebuild)
    status = "done" if result["ok"] else "with-errors"
    print(
        f"embed stage: doc_id={args.doc_id} "
        f"cards_scanned={result['cards_scanned']} "
        f"embedded={result['embedded']} "
        f"skipped={result['skipped']} "
        f"dim={result['dim']} "
        f"failed={len(result['failed'])} "
        f"status={status}"
    )
    return 0 if result["ok"] else 1


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
    p_ocr.add_argument(
        "--backend", choices=("stub", "vlm"), default="stub",
        help="OCR backend: 'stub' (offline placeholder, CI default) or 'vlm' "
             "(UnlimitedOcrBackend — HTTP client of the SGLang server). The VLM "
             "backend reads its endpoint from MASSIVE_PDF_VLM_ENDPOINT (default "
             "http://127.0.0.1:10000/v1); see docs/runbooks/sglang-unlimited-ocr.md "
             "for the launch recipe. Default: stub.",
    )

    p_struct = sub.add_parser(
        "structure",
        help="Parse OCR artifacts into a clause graph (slice 3 stage 3)",
    )
    p_struct.add_argument("doc_id", type=int, help="Document id")
    p_struct.add_argument(
        "--out", default=".massive_pdf", help="Output root for structure artifacts"
    )
    p_struct.add_argument(
        "--rebuild", action="store_true",
        help="Delete existing clauses for this document before parsing",
    )

    p_inv = sub.add_parser(
        "invariants",
        help="Run structural-invariant checks on a document's clause graph",
    )
    p_inv.add_argument("doc_id", type=int, help="Document id")

    p_cards = sub.add_parser(
        "cards",
        help="Run the rule-cards stage (slice 5): derive cards per clause",
    )
    p_cards.add_argument("doc_id", type=int, help="Document id")
    p_cards.add_argument(
        "--out", default=".massive_pdf",
        help="Output root for cards artifacts",
    )
    p_cards.add_argument(
        "--rebuild", action="store_true",
        help="Delete existing cards + checkpoints before re-running",
    )

    p_embed = sub.add_parser(
        "embed",
        help="Run the embed stage (slice 5): encode rule-card statements",
    )
    p_embed.add_argument("doc_id", type=int, help="Document id")
    p_embed.add_argument(
        "--dim", type=int, default=HashBagEncoder.DEFAULT_DIM,
        help="Embedding dim for HashBagEncoder (default 128)",
    )
    p_embed.add_argument(
        "--rebuild", action="store_true",
        help="Recompute every embedding, ignoring checkpoints",
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
        "structure": cmd_structure,
        "invariants": cmd_invariants,
        "cards": cmd_cards,
        "embed": cmd_embed,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
