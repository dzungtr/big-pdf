# ADR 0002: Build the corpus brain, not the agent

## Status
Accepted

## Context
The goal is regulatory reasoning (explain rules, test scenarios, classify actions). Conversational agents (Codex, Claude Code) already interrogate and reason well; rebuilding that is redundant.

## Decision
This system is an **ingestion + retrieval layer only**: OCR → clause graph → rule cards → hybrid retrieval + graph traversal, exposed as a local API + CLI. External LLM agents drive interrogation and verdicts by calling this layer.

## Consequences
- The API contract is the product; its consumer is an LLM, so responses are structured, citation-dense, Vietnamese-original.
- Rule cards carry required-facts checklists so calling agents can interrogate precisely.
- No chat UI, no verdict engine, no session state in this system.
- Trust boundary: this layer supplies ground truth (original clause text + page refs); the agent owns judgment.

## Measured results
(To be filled at initiative close.)
