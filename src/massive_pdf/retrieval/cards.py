"""Rule-card schema + extraction (slice 5, issue #5).

Cards are derived per slice-3 clause. They are *non-authoritative*:
their text comes from an extractor (LLM in production; regex/heuristics
in this stub), and the source of truth is always the clause body
itself. Downstream retrieval cites the card as a hint and the clause
body as the canonical reference.

Schema (the JSON contract the `embed` stage consumes):
  statement        free-form plain-language gloss, ≤ MAX_STATEMENT_CHARS chars
  bound_actor      the party bearing the obligation/right (non-empty)
  topic_tags       list[str] of canonical short tags (e.g. 'hóa đơn', 'thuế suất')
  required_facts   list[str] checklist of facts that must hold for the rule to apply
  penalty          str | None, the consequence if the rule is broken (None if not stated)
"""
from __future__ import annotations
import re
from dataclasses import asdict, dataclass, field

MAX_STATEMENT_CHARS = 600


@dataclass
class RuleCard:
    statement: str
    bound_actor: str
    topic_tags: list[str] = field(default_factory=list)
    required_facts: list[str] = field(default_factory=list)
    penalty: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RuleCardValidationError(ValueError):
    """Raised when a card fails schema validation."""


def validate_card(card: RuleCard) -> None:
    """Validate a RuleCard against the schema. Raises on failure.

    Rules:
      * `statement` must be non-empty after .strip() and ≤ MAX_STATEMENT_CHARS
      * `bound_actor` must be non-empty after .strip()
      * `topic_tags` must be a list of non-empty strings
      * `required_facts` must be a list of non-empty strings
      * `penalty` must be None or a non-empty string
    """
    if not isinstance(card, RuleCard):
        raise RuleCardValidationError(f"expected RuleCard, got {type(card).__name__}")
    if not card.statement or not card.statement.strip():
        raise RuleCardValidationError("statement must be a non-empty string")
    if len(card.statement) > MAX_STATEMENT_CHARS:
        raise RuleCardValidationError(
            f"statement exceeds {MAX_STATEMENT_CHARS} chars (got {len(card.statement)})"
        )
    if not card.bound_actor or not card.bound_actor.strip():
        raise RuleCardValidationError("bound_actor must be a non-empty string")
    if not isinstance(card.topic_tags, list) or not all(
        isinstance(t, str) and t.strip() for t in card.topic_tags
    ):
        raise RuleCardValidationError("topic_tags must be a list of non-empty strings")
    if not isinstance(card.required_facts, list) or not all(
        isinstance(f, str) and f.strip() for f in card.required_facts
    ):
        raise RuleCardValidationError(
            "required_facts must be a list of non-empty strings"
        )
    if card.penalty is not None and (
        not isinstance(card.penalty, str) or not card.penalty.strip()
    ):
        raise RuleCardValidationError(
            "penalty must be a non-empty string or None"
        )


# --- Stub extractor ---------------------------------------------------------
# Without a local LLM we generate one card per top-level (Điều) clause using
# deterministic rules. Khoản/Điểm clauses are refinements of their parent Điều
# and produce no separate card — that keeps the stub's precision/recall
# honest and lets the schema-validation harness exercise real Round-Trips.
# An LLM extractor can swap in later by replacing `extract_cards`.

_DIEU_ACTOR_HINTS: tuple[tuple[str, str], ...] = (
    # Order matters: first hit wins, so list more-specific before generic.
    ("hộ kinh doanh", "hộ kinh doanh"),
    ("cá nhân kinh doanh", "cá nhân kinh doanh"),
    ("cá nhân", "cá nhân"),
    ("doanh nghiệp", "doanh nghiệp"),
    ("tổ chức", "tổ chức"),
)

_TOPIC_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("hóa đơn", "hóa đơn"),
    ("thuế suất", "thuế suất"),
    ("khai thuế", "khai thuế"),
    ("nộp thuế", "nộp thuế"),
    ("miễn thuế", "miễn thuế"),
    ("giảm thuế", "giảm thuế"),
    ("chứng từ", "chứng từ"),
    ("sổ sách", "sổ sách"),
    ("quản lý thuế", "quản lý thuế"),
    ("xử phạt", "xử phạt"),
    ("phạt", "xử phạt"),
)

_PENALTY_RE = re.compile(
    r"phạt\s+[^.\n]{4,200}",
    re.IGNORECASE | re.UNICODE,
)
_DIEU_HEADER_RE = re.compile(r"^\s*Điều\s+\d+\s*[\.:]?\s*", re.UNICODE)


def _strip_header(body: str) -> str:
    """Drop the leading 'Điều N. ' marker so the body starts at the prose."""
    return _DIEU_HEADER_RE.sub("", body, count=1)


def _first_clause(body: str, max_chars: int = 280) -> str:
    text = body.strip().split("\n", 1)[0].strip()
    for stop in (". ", "; "):
        idx = text.find(stop)
        if 0 < idx <= max_chars:
            return text[:idx].strip()
    return text[:max_chars].strip()


def _detect_actor(body: str) -> str:
    bl = body.lower()
    for needle, actor in _DIEU_ACTOR_HINTS:
        if needle in bl:
            return actor
    return "người nộp thuế"


def _detect_topic(body: str) -> str | None:
    bl = body.lower()
    for needle, topic in _TOPIC_KEYWORDS:
        if needle in bl:
            return topic
    return None


def extract_cards(clause_kind: str, clause_citation: str, clause_body: str
                  ) -> list[RuleCard]:
    """Deterministic stub extractor.

    Returns one `RuleCard` for every Điều (top-level) clause whose body is
    non-empty after stripping the header. Khoản/Điểm refinements return
    [] so the parent Điều remains the single authoritative carrier.

    The card is intentionally schema-conservative: every field is
    populated so the validator runs on a realistic record, but the gloss
    is just the first sentence of the body — fine for unit tests, clearly
    insufficient for production retrieval. An LLM extractor replaces
    this function and writes its output through the same schema.
    """
    if clause_kind != "dieu":
        return []
    body = _strip_header(clause_body).strip()
    if not body:
        return []

    gloss = _first_clause(body)
    actor = _detect_actor(body)
    topic = _detect_topic(body)
    penalty_match = _PENALTY_RE.search(body)
    penalty = penalty_match.group(0).strip() if penalty_match else None

    card = RuleCard(
        statement=f"{clause_citation}: {gloss}",
        bound_actor=actor,
        topic_tags=[topic] if topic else [],
        required_facts=[f"Áp dụng theo {clause_citation}"],
        penalty=penalty,
    )
    return [card]
