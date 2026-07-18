# ADR 0003: Clause graph + rule cards + hybrid retrieval

## Status
Accepted

## Context
Legal queries are both semantic ("what binds small enterprises on invoice retention?") and exact-structural ("Điều 12 Khoản 3", "mẫu 01/GTGT"). Pure vector search drops the latter. Regulations cross-reference heavily, including external instruments.

## Decision
1. Parse each Document into a **clause graph** (Điều → Khoản → Điểm) with citations and cross-reference edges.
2. Derive **rule cards** (gloss, actor, topics, required-facts, penalty) — non-authoritative, regenerable; grounding always cites original clause text.
3. Retrieval contract: **hybrid** `retrieve(query, structural_filters)` **plus** full graph traversal (`get_clause`, children/parents, references/referenced-by).
4. External references **auto-ingest on demand** (recursive corpus growth); fetched docs record source URL + version/effective date.

## Consequences
- Cross-reference extraction must handle Vietnamese patterns ("khoản 2 Điều này", "theo Điều 5", "theo Thông tư số ...").
- Auto-ingest creates a version-currency risk; every fetched document must carry provenance (source, fetched-at, effective date) and the graph must tolerate amendments superseding nodes.
- Two-query pattern (retrieve → expand via graph) is the expected agent flow.

## Measured results
(To be filled at initiative close: retrieval precision@k on a hand-labeled query set; cross-reference extraction precision/recall on sampled clauses.)
