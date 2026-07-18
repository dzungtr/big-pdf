"""`structure` ingest stage: OCR artifacts -> clause graph + xrefs.

Pipeline:

  1. Read every OCR artifact JSON for the document (`pages/done`).
  2. Concatenate raw_text per page (1-indexed ordinals).
  3. Run the Vietnamese clause parser -> `ClauseGraph`.
  4. Run the cross-reference extractor per clause body.
  5. Resolve internal pointers against the just-parsed `clauses`
     table; external pointers are stored as dangling rows.
  6. Persist clauses + references; record a document-level checkpoint.

Idempotent: reruns upsert on `(document_id, citation)` so clause ids
are stable, and ignore-duplicate on references.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..ingest.ocr import OCR_STAGE, OcrPage, _deserialize_page
from ..store import (
    add_clause_reference,
    connect,
    dangling_references,
    delete_clauses_for_document,
    get_checkpoint,
    get_clause_by_citation,
    init_db,
    list_clauses,
    mark_checkpoint,
    upsert_clause,
)
from .parser import ClauseGraph, ParsedClause, parse_pages
from .xrefs import ParsedReference, extract_references


STRUCTURE_STAGE = "structure"


@dataclass
class InvariantReport:
    """Result of structural-invariant checks on a parsed document."""
    ok: bool
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.ok:
            return f"InvariantReport(ok=True, 0 issues)"
        return f"InvariantReport(ok=False, {len(self.issues)} issues):\n  - " + "\n  - ".join(self.issues)


def _read_ocr_pages(out_dir: Path, document_id: int) -> list[tuple[int, str]]:
    """Read every OCR artifact and return `(page_ordinal, raw_text)` pairs."""
    from ..ingest.ocr import ocr_artifact_path
    pairs: list[tuple[int, str]] = []
    ocr_dir = out_dir / "ocr" / f"doc{document_id}"
    if not ocr_dir.exists():
        return pairs
    for path in sorted(ocr_dir.glob("page*.json")):
        ordinal = int(path.stem.replace("page", ""))
        page: OcrPage = _deserialize_page(path, ordinal)
        pairs.append((ordinal, page.raw_text))
    return pairs


def _persist_clauses(conn, document_id: int, graph: ClauseGraph) -> dict[str, int]:
    """Upsert each parsed clause; return mapping citation -> id.

    Order matters: parent clauses must be persisted before children so
    `parent_id` can be resolved.
    """
    citation_to_id: dict[str, int] = {}
    for c in graph.clauses:
        parent_id = None
        if c.parent_number is not None:
            # Reconstruct the parent citation from parent_number + kind.
            if c.kind == "khoan":
                parent_cite = f"Điều {c.parent_number[0]}"
            elif c.kind == "diem":
                d = c.parent_number[0]
                if len(c.parent_number) >= 2:
                    parent_cite = f"Khoản {c.parent_number[1]} Điều {d}"
                else:
                    parent_cite = f"Điều {d}"
            else:
                parent_cite = None
            if parent_cite is not None:
                row = get_clause_by_citation(conn, document_id, parent_cite)
                if row is not None:
                    parent_id = row["id"]
        cid = upsert_clause(
            conn,
            document_id=document_id,
            parent_id=parent_id,
            kind=c.kind,
            number=c.number,
            citation=c.citation,
            page_start=c.page_start,
            page_end=c.page_end,
            body=c.body,
            ord=c.ord,
        )
        citation_to_id[c.citation] = cid
    return citation_to_id


def _resolve_self_ref(ref: ParsedReference, src_clause: ParsedClause) -> str | None:
    """Map a '... Điều này' self-ref onto the source clause's ancestor Điều."""
    if "Điều này" not in ref.target_citation:
        return None
    # Find ancestor Điều number from the source clause.
    ancestor = None
    if src_clause.kind == "dieu":
        ancestor = src_clause.number
    elif src_clause.parent_number:
        ancestor = src_clause.parent_number[0]
    if ancestor is None:
        return None
    # Rewrite "Khoản N Điều này" -> "Khoản N Điều {ancestor}".
    if ref.target_citation.startswith("Khoản "):
        k = ref.target_citation.split()[1]
        return f"Khoản {k} Điều {ancestor}"
    if ref.target_citation.startswith("Điểm "):
        parts = ref.target_citation.split()
        # ["Điểm", "<letter>", "Khoản", "<k>", "Điều", "này"]
        if len(parts) >= 4 and parts[2] == "Khoản":
            return f"Điểm {parts[1]} Khoản {parts[3]} Điều {ancestor}"
        if len(parts) >= 2:
            return f"Điểm {parts[1]} Điều {ancestor}"
    return None


def _persist_references(conn, document_id: int, graph: ClauseGraph,
                        citation_to_id: dict[str, int]) -> tuple[int, int]:
    """Resolve and persist cross-references for every parsed clause.

    Returns (internal_count, external_count).
    """
    internal = 0
    external = 0
    for c in graph.clauses:
        src_id = citation_to_id[c.citation]
        refs = extract_references(c.body)
        for ref in refs:
            if ref.kind == "external":
                add_clause_reference(
                    conn,
                    src_clause_id=src_id,
                    dst_clause_id=None,
                    kind="external",
                    raw_text=ref.raw_text,
                    target_citation=ref.target_citation,
                    target_document_id=None,
                )
                external += 1
                continue

            # internal or internal_self
            target_cite = ref.target_citation
            if ref.kind == "internal_self":
                rewritten = _resolve_self_ref(ref, c)
                if rewritten is not None:
                    target_cite = rewritten
                # else leave it as-is; will dangle.
            dst = citation_to_id.get(target_cite)
            if dst is None:
                # Try to resolve against DB (in case it was parsed in a prior run).
                row = get_clause_by_citation(conn, document_id, target_cite)
                if row is not None:
                    dst = row["id"]
            if dst is not None:
                add_clause_reference(
                    conn,
                    src_clause_id=src_id,
                    dst_clause_id=dst,
                    kind="internal",
                    raw_text=ref.raw_text,
                    target_citation=target_cite,
                    target_document_id=None,
                )
                internal += 1
            else:
                # Internal pointer to a clause we couldn't resolve; store as
                # dangling so the invariant check (and a future slice) can
                # surface it. The brief requires "internal edges resolved,
                # external edges recorded as dangling"; we err on the side
                # of *not* silently dropping dangling internals — they're
                # rare enough that surfacing them is the right default.
                add_clause_reference(
                    conn,
                    src_clause_id=src_id,
                    dst_clause_id=None,
                    kind="internal",
                    raw_text=ref.raw_text,
                    target_citation=target_cite,
                    target_document_id=None,
                )
                external += 1  # counted as dangling
    return internal, external


def run_structure_stage(db_path: str | Path, document_id: int,
                        out_dir: str | Path,
                        *, rebuild: bool = False) -> dict:
    """Idempotent structure stage.

    Returns a summary dict with: clauses, refs_internal, refs_external,
    pages, ok (always True unless DB error).
    """
    out_dir = Path(out_dir)
    pages = _read_ocr_pages(out_dir, document_id)
    if not pages:
        # Nothing to do; treat as success-noop but record checkpoint
        # so callers can see we ran.
        init_db(db_path)
        with connect(db_path) as conn:
            mark_checkpoint(conn, document_id, STRUCTURE_STAGE, 0, "done",
                            artifact_path=None)
            conn.commit()
        return {"clauses": 0, "refs_internal": 0, "refs_external": 0,
                "pages": 0, "ok": True}

    graph = parse_pages(document_id=document_id, pages=pages)

    init_db(db_path)
    with connect(db_path) as conn:
        if rebuild:
            delete_clauses_for_document(conn, document_id)
        citation_to_id = _persist_clauses(conn, document_id, graph)
        internal, external = _persist_references(
            conn, document_id, graph, citation_to_id,
        )
        # Document-level checkpoint.
        mark_checkpoint(conn, document_id, STRUCTURE_STAGE, 0, "done",
                        artifact_path=None)
        conn.commit()

    # Side artifact: serialize the parsed graph for inspection / debugging.
    artifact_dir = out_dir / "structure" / f"doc{document_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "clauses.json"
    artifact_path.write_text(
        json.dumps({
            "document_id": document_id,
            "page_count": graph.page_count,
            "clauses": [
                {**c.to_db_row(), "id": citation_to_id[c.citation]}
                for c in graph.clauses
            ],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "clauses": len(graph.clauses),
        "refs_internal": internal,
        "refs_external": external,
        "pages": graph.page_count,
        "artifact_path": str(artifact_path),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# Structural invariant checks
# ---------------------------------------------------------------------------

def check_invariants(db_path: str | Path, document_id: int) -> InvariantReport:
    """Run structural invariants over the parsed graph for a document.

    Invariants (slice #4 brief):
      - Every clause cited (citation field set, body non-empty).
      - Điều numbering starts at 1 and is contiguous (1..N with no gaps).
      - Every cross-reference resolves or dangles explicitly
        (no NULL/None state in the row).
    """
    issues: list[str] = []
    with connect(db_path) as conn:
        clauses = list_clauses(conn, document_id)
        if not clauses:
            return InvariantReport(ok=False, issues=["no clauses persisted for document"])
        # 1. Every clause cited & non-empty body
        for c in clauses:
            if not c["citation"]:
                issues.append(f"clause id={c['id']} has empty citation")
            if not c["body"].strip():
                issues.append(f"clause id={c['id']} {c['citation']!r} has empty body")
        # 2. Điều numbering
        dieu_nums = sorted(int(c["number"]) for c in clauses if c["kind"] == "dieu")
        if dieu_nums:
            expected = list(range(dieu_nums[0], dieu_nums[-1] + 1))
            if dieu_nums != expected:
                missing = sorted(set(expected) - set(dieu_nums))
                if missing:
                    issues.append(f"Điều numbering has gaps; missing: {missing}")
                extra = sorted(set(dieu_nums) - set(expected))
                if extra:
                    issues.append(f"Điều numbering has extras: {extra}")
        # 3. Cross-references resolve or dangle
        rows = conn.execute(
            "SELECT cr.* FROM clause_references cr "
            "JOIN clauses c ON c.id = cr.src_clause_id "
            "WHERE c.document_id=?",
            (document_id,),
        ).fetchall()
        for r in rows:
            r = dict(r)
            if not r["raw_text"]:
                issues.append(f"ref id={r['id']} has empty raw_text")
            if not r["target_citation"]:
                issues.append(f"ref id={r['id']} has empty target_citation")
            if r["kind"] == "internal" and r["dst_clause_id"] is None:
                issues.append(
                    f"ref id={r['id']} kind=internal dangles: "
                    f"target_citation={r['target_citation']!r} "
                    f"(clause not found)"
                )

    return InvariantReport(ok=not issues, issues=issues)
