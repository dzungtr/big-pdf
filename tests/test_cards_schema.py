"""Schema-validation tests for the slice-5 RuleCard dataclass."""
import pytest

from massive_pdf.retrieval.cards import (
    MAX_STATEMENT_CHARS,
    RuleCard,
    RuleCardValidationError,
    validate_card,
)


def test_valid_card_passes():
    card = RuleCard(
        statement="Điều 1: Hộ kinh doanh kê khai thuế theo quý.",
        bound_actor="hộ kinh doanh",
        topic_tags=["khai thuế"],
        required_facts=["Có đăng ký kinh doanh"],
        penalty=None,
    )
    validate_card(card)  # no raise


def test_empty_statement_rejected():
    card = RuleCard(statement="  ", bound_actor="hộ kinh doanh")
    with pytest.raises(RuleCardValidationError, match="statement"):
        validate_card(card)


def test_statement_too_long_rejected():
    card = RuleCard(
        statement="x" * (MAX_STATEMENT_CHARS + 1),
        bound_actor="hộ kinh doanh",
    )
    with pytest.raises(RuleCardValidationError, match="exceeds"):
        validate_card(card)


def test_empty_bound_actor_rejected():
    card = RuleCard(statement="x", bound_actor="")
    with pytest.raises(RuleCardValidationError, match="bound_actor"):
        validate_card(card)


def test_topic_tags_must_be_list_of_strings():
    card = RuleCard(statement="x", bound_actor="y", topic_tags=["a", ""])
    with pytest.raises(RuleCardValidationError, match="topic_tags"):
        validate_card(card)


def test_required_facts_must_be_list_of_strings():
    card = RuleCard(statement="x", bound_actor="y", required_facts=[1, 2])
    with pytest.raises(RuleCardValidationError, match="required_facts"):
        validate_card(card)


def test_penalty_must_be_string_or_none():
    card = RuleCard(statement="x", bound_actor="y", penalty="")
    with pytest.raises(RuleCardValidationError, match="penalty"):
        validate_card(card)
    card2 = RuleCard(statement="x", bound_actor="y", penalty="Phạt cảnh cáo")
    validate_card(card2)  # ok
    card3 = RuleCard(statement="x", bound_actor="y", penalty=None)
    validate_card(card3)  # also ok


def test_to_dict_roundtrip():
    card = RuleCard(
        statement="Điều 1: Hộ kinh doanh kê khai thuế theo quý.",
        bound_actor="hộ kinh doanh",
        topic_tags=["khai thuế"],
        required_facts=["Có đăng ký kinh doanh"],
        penalty=None,
    )
    d = card.to_dict()
    assert d["statement"] == card.statement
    assert d["bound_actor"] == card.bound_actor
    assert d["topic_tags"] == ["khai thuế"]
    assert d["required_facts"] == ["Có đăng ký kinh doanh"]
    assert d["penalty"] is None


def test_validate_rejects_non_rulecard():
    with pytest.raises(RuleCardValidationError, match="RuleCard"):
        validate_card({"statement": "x", "bound_actor": "y"})  # type: ignore[arg-type]
