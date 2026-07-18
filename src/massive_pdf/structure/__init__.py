"""Structure stage: clause-graph parsing from OCR output (slice 3, issue #4).

Public API:
  - `parse_pages(document_id, pages)` -> `ClauseGraph`
  - `extract_references(body)`       -> `list[ParsedReference]`
  - `run_structure_stage(...)`        -> dict summary
  - `check_invariants(...)`           -> `InvariantReport`
"""
from .parser import (
    ClauseGraph,
    ParsedClause,
    parse_pages,
)
from .xrefs import (
    ParsedReference,
    extract_references,
)
from .stage import (
    STRUCTURE_STAGE,
    InvariantReport,
    check_invariants,
    run_structure_stage,
)

__all__ = [
    "ParsedClause",
    "ClauseGraph",
    "parse_pages",
    "ParsedReference",
    "extract_references",
    "STRUCTURE_STAGE",
    "InvariantReport",
    "check_invariants",
    "run_structure_stage",
]
