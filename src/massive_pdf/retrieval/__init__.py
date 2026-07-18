"""Retrieval stages (slice 5, issue #5): rule cards + embeddings.

Public API:
  - `RuleCard`, `validate_card`, `extract_cards`
  - `Encoder`, `HashBagEncoder`, `BgeM3Encoder`, `get_default_encoder`
  - `encode_embedding`, `decode_embedding`
  - `run_cards_stage`, `run_embed_stage`, `CARDS_STAGE`, `EMBED_STAGE`
"""
from .cards import (
    MAX_STATEMENT_CHARS,
    RuleCard,
    RuleCardValidationError,
    extract_cards,
    validate_card,
)
from .encoder import (
    BgeM3Encoder,
    Encoder,
    HashBagEncoder,
    decode_embedding,
    encode_embedding,
    get_default_encoder,
    normalize,
    tokenize,
)
from .stage import (
    CARDS_STAGE,
    EMBED_STAGE,
    check_card_dimensions,
    run_cards_stage,
    run_embed_stage,
)

__all__ = [
    # schema / extraction
    "MAX_STATEMENT_CHARS",
    "RuleCard",
    "RuleCardValidationError",
    "extract_cards",
    "validate_card",
    # encoder
    "BgeM3Encoder",
    "Encoder",
    "HashBagEncoder",
    "decode_embedding",
    "encode_embedding",
    "get_default_encoder",
    "normalize",
    "tokenize",
    # stages
    "CARDS_STAGE",
    "EMBED_STAGE",
    "check_card_dimensions",
    "run_cards_stage",
    "run_embed_stage",
]
