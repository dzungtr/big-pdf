# ADR 0004: SQLite + sqlite-vec for the MVP store

## Status
Accepted

## Context
Three access patterns: graph traversal, vector search, structural filtering. Corpus scale is ~10⁴–10⁵ clauses total across dozens of documents — small by DB standards. Single-user, local, write-rarely/read-mostly.

## Decision
SQLite with the `sqlite-vec` extension: relational tables for documents/clauses/references/rule-cards, vector table for embeddings, recursive CTEs for traversal, JSON columns for flexible card fields.

## Alternatives
- **Postgres + pgvector** — heavier ops burden than the MVP warrants; migration path later is mechanical.
- **Graph DB (Neo4j/Kuzu)** — unnecessary at this graph size.

## Consequences
- Corpus is a single inspectable file; backup = copy.
- Concurrency is not a concern at MVP; if the API later serves concurrent writers, revisit.

## Measured results
(To be filled at initiative close: query latency on traversal + vector search at corpus scale.)
