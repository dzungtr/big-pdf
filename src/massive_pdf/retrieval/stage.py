"""`cards` + `embed` ingest stages (slice 5, issue #5).

`cards`:  for each clause in a document, derive ≥0 `RuleCard`s via the
  extractor and persist them into the `rule_cards` table linked to
  `clauses.id`.

`embed`:  compute a normalised vector per rule card (statement + clause
  body) and write it back to `rule_cards.embedding`.

Both stages are idempotent and per-clause resumable via
`ingest_checkpoints`: we use `page_ordinal=clause_id` for per-clause
markers and `page_ordinal=0` for the document-level "all done" marker
(matching the convention established in slices 2 + 3).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..store import (
    completed_page_ordinals,
    connect,
    count_rule_cards_by_document,
    get_clause,
    init_db,
    insert_rule_card,
    list_clauses,
    list_rule_cards_for_document,
    mark_checkpoint,
    update_rule_card_embedding,
)
from .cards import RuleCard, RuleCardValidationError, extract_cards, validate_card
from .encoder import Encoder, encode_embedding


CARDS_STAGE = "cards"
EMBED_STAGE = "embed"


@dataclass
class CardsStageResult:
    document_id: int
    clauses_scanned: int = 0
    cards_inserted: int = 0
    skipped: int = 0
    failed: list[dict] = field(default_factory=list)
    artifact_path: str | None = None

    @property
    def ok(self) -> bool:
        return not self.failed


@dataclass
class EmbedStageResult:
    document_id: int
    cards_scanned: int = 0
    embedded: int = 0
    skipped: int = 0
    dim: int = 0
    mismatched: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def _checkpoints_for(conn, document_id: int, stage: str) -> set[int]:
    """Re-export to keep the import surface local."""
    return completed_page_ordinals(conn, document_id, stage)


def _row_inserted_ids(conn, *, clause_id: int, statement: str) -> int:
    return insert_rule_card(conn, clause_id=clause_id,
                            statement=statement, embedding=None)


def run_cards_stage(db_path: str | Path, document_id: int,
                    out_dir: str | Path,
                    *,
                    rebuild: bool = False) -> dict[str, Any]:
    """Idempotent `cards` stage. Returns a JSON-serialisable summary dict.

    Behaviour:
      * Walks every clause returned by `list_clauses(document_id)`.
      * For each clause whose checkpoint is not yet `done` (or with
        `rebuild=True`), runs `extract_cards` + `validate_card` and
        inserts one row per card via `insert_rule_card`.
      * Per-clause failures (validation errors) are captured in `failed`
        and the clause's checkpoint is marked `failed`; the stage does
        NOT abort.
      * Writes a `cards.json` artifact under `out_dir/cards/doc<id>/`.
    """
    init_db(db_path)
    out_dir = Path(out_dir)
    result = CardsStageResult(document_id=document_id)

    with connect(db_path) as conn:
        if rebuild:
            conn.execute(
                """
                DELETE FROM rule_cards
                WHERE clause_id IN (SELECT id FROM clauses WHERE document_id=?)
                """,
                (document_id,),
            )
            conn.execute(
                """
                DELETE FROM ingest_checkpoints
                WHERE document_id=? AND stage IN (?, ?)
                """,
                (document_id, CARDS_STAGE, EMBED_STAGE),
            )

        clauses = list_clauses(conn, document_id)
        result.clauses_scanned = len(clauses)
        already_done = completed_page_ordinals(conn, document_id, CARDS_STAGE)

        for c in clauses:
            cid = c["id"]
            if not rebuild and cid in already_done:
                result.skipped += 1
                continue
            try:
                cards = extract_cards(c["kind"], c["citation"], c["body"])
                for card in cards:
                    validate_card(card)
                    # Only the gloss goes into the per-row `statement` column;
                    # the full card (incl. actor / facts) is recoverable from
                    # the artifact file written below.
                    insert_rule_card(
                        conn,
                        clause_id=cid,
                        statement=card.statement,
                        embedding=None,
                    )
                    result.cards_inserted += 1
                mark_checkpoint(conn, document_id, CARDS_STAGE, cid, "done")
            except RuleCardValidationError as e:
                mark_checkpoint(conn, document_id, CARDS_STAGE, cid, "failed")
                result.failed.append({"clause_id": cid, "error": str(e)})

        # Document-level "stage ran" marker (per existing convention).
        mark_checkpoint(conn, document_id, CARDS_STAGE, 0, "done")
        conn.commit()

    # Side artifact — plain JSON for inspection. Embeddings are
    # serialised as hex so the JSON is portable.
    artifact_dir = out_dir / "cards" / f"doc{document_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "cards.json"
    with connect(db_path) as conn:
        rows = list_rule_cards_for_document(conn, document_id)
    artifact_path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "card_count": len(rows),
                "cards": [
                    {
                        "id": r["id"],
                        "clause_id": r["clause_id"],
                        "statement": r["statement"],
                        "has_embedding": r["embedding"] is not None,
                    }
                    for r in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    result.artifact_path = str(artifact_path)

    return {
        "document_id": result.document_id,
        "clauses_scanned": result.clauses_scanned,
        "cards_inserted": result.cards_inserted,
        "skipped": result.skipped,
        "failed": result.failed,
        "artifact_path": result.artifact_path,
        "ok": result.ok,
    }


def run_embed_stage(db_path: str | Path, document_id: int,
                    encoder: Encoder,
                    *,
                    rebuild: bool = False) -> dict[str, Any]:
    """Idempotent `embed` stage. Returns a JSON-serialisable summary dict.

    Behaviour:
      * Walks every rule card for the document.
      * Encodes `statement + clause_body` and writes the float32 bytes
        into `rule_cards.embedding`.
      * Skips cards whose checkpoint is already `done` for the `embed`
        stage and whose existing embedding matches `encoder.dim`.
      * Dimension mismatches are recorded in `mismatched` and the row's
        checkpoint is left untouched so a re-run with the right model
        can fill it in.
    """
    init_db(db_path)
    dim = encoder.dim
    result = EmbedStageResult(document_id=document_id, dim=dim)

    with connect(db_path) as conn:
        cards = list_rule_cards_for_document(conn, document_id)
        result.cards_scanned = len(cards)
        done_ordinals = completed_page_ordinals(conn, document_id, EMBED_STAGE)

        for row in cards:
            cid = row["clause_id"]
            existing = row["embedding"]
            if (
                not rebuild
                and cid in done_ordinals
                and existing is not None
                and len(existing) == dim * 4  # float32 = 4 bytes/elem
            ):
                result.skipped += 1
                continue
            try:
                text = _compose_embed_text(conn, row)
                vec = np.asarray(encoder.encode(text), dtype=np.float32)
                if vec.ndim != 1:
                    raise ValueError(
                        f"encoder returned non-1-D vector: shape={vec.shape}"
                    )
                if vec.shape[0] != dim:
                    raise ValueError(
                        f"encoder dim mismatch: produced {vec.shape[0]}, "
                        f"encoder.dim={dim}"
                    )
                blob = encode_embedding(vec)
                update_rule_card_embedding(conn, row["id"], blob)
                mark_checkpoint(conn, document_id, EMBED_STAGE, cid, "done")
                result.embedded += 1
            except Exception as e:  # noqa: BLE001
                mark_checkpoint(conn, document_id, EMBED_STAGE, cid, "failed")
                result.failed.append({"rule_card_id": row["id"], "error": str(e)})

        mark_checkpoint(conn, document_id, EMBED_STAGE, 0, "done")
        conn.commit()

    return {
        "document_id": result.document_id,
        "cards_scanned": result.cards_scanned,
        "embedded": result.embedded,
        "skipped": result.skipped,
        "dim": result.dim,
        "mismatched": result.mismatched,
        "failed": result.failed,
        "ok": result.ok,
    }


def check_card_dimensions(db_path: str | Path, document_id: int,
                          expected_dim: int) -> dict[str, Any]:
    """Sanity-check that every card's embedding matches `expected_dim`.

    Returns a dict with `ok`, `checked`, `mismatched`.
    """
    with connect(db_path) as conn:
        rows = list_rule_cards_for_document(conn, document_id)
    checked = 0
    mismatched: list[dict] = []
    for r in rows:
        emb = r["embedding"]
        if emb is None:
            continue
        actual = len(emb) // 4  # float32 = 4 bytes
        checked += 1
        if actual != expected_dim:
            mismatched.append({"id": r["id"], "actual_dim": actual,
                               "expected_dim": expected_dim})
    return {"ok": not mismatched, "checked": checked,
            "mismatched": mismatched, "expected_dim": expected_dim}


def _compose_embed_text(conn, rule_card_row: dict) -> str:
    """Compose the text fed to the encoder: gloss + source clause body."""
    clause = get_clause(conn, rule_card_row["clause_id"]) or {}
    body = clause.get("body", "") or ""
    citation = clause.get("citation", "") or ""
    return (
        f"{rule_card_row['statement']}\n\n"
        f"Source: {citation}\n{body}".strip()
    )
