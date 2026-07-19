"""`ocr-import` stage: Unlimited-OCR markdown -> per-page OcrPage artifacts.

Unlimited-OCR's native behavior is to convert an entire PDF into one
markdown file in a single long-horizon pass (see
docs/runbooks/sglang-unlimited-ocr.md). This stage consumes that file and
splits it back into the per-page `OcrPage` JSON artifacts the rest of the
pipeline expects (the `structure` stage reads `<out>/ocr/doc<id>/page*.json`),
so a document can be ingested without re-running OCR page-by-page.

Split strategy
--------------
The markdown carries two kinds of page boundary:

1. **Batch markers** ``<!-- page N-M -->`` — written by the project's
   ingestion script, one per multi-image request, covering exactly the
   page ordinals N..M. These are authoritative: they always exist and
   always match the PDF's 1-indexed page ordinals.

2. **``<PAGE>`` markers** emitted by the model, each followed by a
   ``<|det|>page_number [...]<|/det|><printed-number>`` line. These are
   *not* 1:1 with PDF pages: the model omits them on blank/cover pages,
   sometimes emits duplicates, and in appendices the printed number
   restarts and no longer equals the PDF ordinal. They cannot be trusted
   as page boundaries.

So this stage splits each batch's body into one `OcrPage` per ordinal in
N..M by walking the batch's `<PAGE>` markers in order and assigning them
to successive ordinals; any trailing ordinals in the batch with no
matching marker get an empty `OcrPage` (a blank/cover page). This keeps
clause *bodies* intact (the `structure` parser concatenates per-page
`raw_text` and finds headers across the whole text) and gives best-effort
`page_start`/`page_end` provenance.

Idempotent + resumable like the other stages: pages already marked `done`
for the `ocr` stage are skipped.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..store import (
    completed_page_ordinals,
    connect,
    list_pages,
    mark_checkpoint,
)
from .ocr import (
    OCR_STAGE,
    OcrPage,
    _serialize_page,
    ocr_artifact_path,
)

# Authoritative batch marker written by the ingestion script: <!-- page N-M -->
_BATCH_RE = re.compile(r"(?m)^<!-- page (\d+)-(\d+) -->\s*$")
# Model-emitted page marker. A <PAGE> always begins a new page's content.
_PAGE_RE = re.compile(r"(?m)^<PAGE>")
# Layout markup prefix the VLM emits on every line: <|det|>kind [bbox]<|/det|>
# e.g. "<|det|>title [198,670,471,691]<|/det|>Điều 2. Đối tượng áp dụng".
# The prefix carries layout metadata the structure parser and embedder don't
# use; stripping it leaves clean Vietnamese text with headers at line start.
_DET_RE = re.compile(r"(?m)^<\|det\|>[^<]*<\|/det\|>")
# The first det after <PAGE> is always the printed page_number; not clause
# content, so drop the whole line.
_PAGE_NUMBER_LINE_RE = re.compile(r"(?m)^<\|det\|>page_number\b[^<]*<\|/det\|>\s*\d+\s*$\n?", re.M)
_NON_TEXT_RE = re.compile(r"(?m)^[ \t]*\[Non-Text\][ \t]*$\n?", re.M)


def _clean_page_text(text: str) -> str:
    """Strip Unlimited-OCR layout markup, leaving clean clause text.

    Converts lines like
        ``<|det|>title [198,670,471,691]<|/det|>Điều 2. Đối tượng áp dụng``
    to
        ``Điều 2. Đối tượng áp dụng``
    and drops the printed page-number line and ``[Non-Text]`` placeholders.
    """
    # Order: drop the page_number line (still has its prefix), strip the det
    # prefix from every line, then drop bare [Non-Text] placeholders left behind.
    text = _PAGE_NUMBER_LINE_RE.sub("", text)
    text = _DET_RE.sub("", text)
    text = _NON_TEXT_RE.sub("", text)
    return text.strip("\n")


def _split_batches(md_text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, body)`` per batch marker, in document order."""
    matches = list(_BATCH_RE.finditer(md_text))
    if not matches:
        raise ValueError(
            "no '<!-- page N-M -->' batch markers found in markdown; "
            "the file is not an Unlimited-OCR ingestion output produced "
            "by this project's runbook"
        )
    batches: list[tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        start, end = int(m.group(1)), int(m.group(2))
        body_begin = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        batches.append((start, end, md_text[body_begin:body_end]))
    return batches


def _split_batch_into_pages(start: int, end: int, body: str) -> dict[int, str]:
    """Map a batch's body to ``{page_ordinal: text}`` for ordinals start..end.

    Walks the batch's ``<PAGE>`` markers in order and assigns each to the
    next ordinal in start..end. Ordinals with no marker (blank/cover pages,
    or a batch that produced fewer markers than pages) get an empty string.
    """
    # Split on <PAGE>; the first chunk before any <PAGE> is preamble
    # (whitespace the ingestion script left) — drop it.
    parts = _PAGE_RE.split(body)
    page_chunks = parts[1:]  # parts[0] is the preamble before the first <PAGE>

    mapping: dict[int, str] = {ordinal: "" for ordinal in range(start, end + 1)}
    ordinals = list(range(start, end + 1))
    for chunk, ordinal in zip(page_chunks, ordinals):
        mapping[ordinal] = _clean_page_text(chunk)
    return mapping


def import_ocr_markdown(
    db_path: str | Path,
    document_id: int,
    md_path: str | Path,
    out_dir: str | Path,
) -> int:
    """Import an Unlimited-OCR markdown file as the `ocr` stage's output.

    Writes one ``OcrPage`` JSON per page ordinal the `pages` stage
    registered for this document and marks each ``ocr`` checkpoint
    ``done``. Resumable: ordinals already ``done`` are skipped.

    Returns the number of page artifacts now present for the document.
    """
    md_path = Path(md_path)
    out_dir = Path(out_dir)
    md_text = md_path.read_text(encoding="utf-8")

    with connect(db_path) as conn:
        done = completed_page_ordinals(conn, document_id, OCR_STAGE)
        known = {p["page_ordinal"] for p in list_pages(conn, document_id)}
        if not known:
            raise ValueError(
                f"document id={document_id} has no pages; run the `pages` "
                f"stage before `ocr-import`"
            )
        image_by_ordinal = {
            p["page_ordinal"]: p["image_path"] for p in list_pages(conn, document_id)
        }

    per_page: dict[int, str] = {}
    for start, end, body in _split_batches(md_text):
        per_page.update(_split_batch_into_pages(start, end, body))

    # Only emit artifacts for ordinals the pages stage registered. Any
    # markdown page outside the registered set (shouldn't happen with the
    # project's own ingestion script) is dropped with a warning.
    written = 0
    with connect(db_path) as conn:
        for ordinal in sorted(known):
            if ordinal in done:
                if Path(image_by_ordinal[ordinal]).exists():
                    written += 1
                continue
            text = per_page.get(ordinal, "")
            artifact = ocr_artifact_path(out_dir, document_id, ordinal)
            page = OcrPage(
                page_ordinal=ordinal,
                image_path=image_by_ordinal[ordinal],
                blocks=[],
                raw_text=text,
            )
            _serialize_page(page, artifact)
            mark_checkpoint(
                conn, document_id, OCR_STAGE, ordinal, "done",
                artifact_path=str(artifact),
            )
            written += 1
        conn.commit()
    return written


__all__ = ["import_ocr_markdown"]
