# Atlas Architecture

This document describes the system as implemented on `main` — every claim
is traceable to a source file listed inline. For milestone history see
[ROADMAP.md](ROADMAP.md); for measured numbers see
[BENCHMARKS.md](BENCHMARKS.md).

## Crate map

| crate | responsibility | depends on |
|---|---|---|
| `atlas-cli` (`apps/`) | clap CLI: `learn`/`search`/`ask` (live), `similar`/`graph` (placeholders) | core, embedding, qa, storage, tokenizer |
| `atlas-core` | figment configuration (`Config`), `AtlasError` | figment, serde, thiserror |
| `atlas-tokenizer` | byte-level BPE: pre-tokenization, ranks parsing, training, encoding (naive + heap engines), decoding | fancy-regex |
| `atlas-vocab` | token → dense stable IDs, deterministic build, versioned bincode persistence | tokenizer |
| `atlas-embedding` | chunking, `Embedder` trait + FastEmbed backend, Skip-Gram learning, vector cache | tokenizer, vocab, fastembed |
| `atlas-graph` | entities, directed relationships with evidence, extraction, BFS, personalized PageRank — **std-only, zero Atlas deps** | — |
| `atlas-storage` | `VectorIndex`/`BruteForceIndex`, SurrealDB store, learn/search workflows, hybrid retrieval | embedding, graph, tokenizer, surrealdb, sha2 |
| `atlas-qa` | extractive question answering with citations and confidence | storage, embedding, graph |

Dependency direction is strictly layered: `graph` and `tokenizer` are
leaves; `storage` composes embedding + graph + tokenizer; `qa` composes
storage + graph. No cycles; `atlas-graph` never touches storage.

## Deterministic identity rules

All identities are content- or first-occurrence-derived, so the same input
always produces the same IDs (`atlas-storage/src/records.rs`,
`atlas-tokenizer/src/ranks.rs`, `atlas-vocab`, `atlas-graph/src/graph.rs`):

- **BPE tokens** are raw byte strings; ranks come from the merge table.
- **Vocabulary IDs** (`atlas-vocab`) are dense `u32`s assigned by first
  occurrence during corpus ingestion; the vocabulary persists as a
  versioned bincode file.
- **`doc_id`** = SHA-256 of the document text (hex).
- **`chunk_id`** = `{doc_id}:{ordinal}` — stable because M02 chunking is
  deterministic (fixed pre-token boundaries).
- **Embedding key** = `(chunk_id, model_id)`; vectors from different
  models never mix.
- **Graph entities** are normalized (lowercased) surface terms with dense
  first-seen IDs and first-seen display labels; **relationships** are
  deduplicated per `(src, predicate, dst)` triple.

## Data flows

### Learn (M07) — `atlas-storage/src/workflow.rs`

```
text → SHA-256 doc_id → Chunker (≤128 tokens, pre-token aligned)
     → one batched FastEmbed call → document/chunk/embedding upserts
     → per-chunk graph extraction (evidence doc_id = chunk_id)
     → graph persistence
```

All writes are idempotent upserts — there are no transactions, by design.
Duplicate documents (complete chunk set present) short-circuit; partial
states (document row without all chunks, e.g. after a crash) are repaired
by the next `learn`.

### Search (M07/M08) — `atlas-storage/src/workflow.rs` + `hybrid.rs`

```
query → embed_query → load embedding table → rebuild BruteForceIndex
     → exact top-k (cosine via L2-normalized dot products)
     → [hybrid] seed entities from chunk-keyed evidence
     → bounded expansion → personalized PageRank
     → candidate chunks ranked by incident PPR mass
     → reciprocal-rank fusion (constant 60) → top-k results with provenance
```

The exact scan itself is microseconds at Atlas scale (see
[BENCHMARKS.md](BENCHMARKS.md)); the dominant durable-search cost is
reloading embeddings and rebuilding the index per query — a documented
M07 simplicity decision.

### Ask (M09) — `atlas-qa/src/engine.rs`

```
question → hybrid retrieval → sentence split + question-word-overlap scoring
        → verbatim answer sentences with [doc, chunk] citations
        → confidence (High/Medium/Low) ← retrieval score + overlap
        → graph lead-in for relationship-shaped questions
```

Empty retrieval or zero overlap ⇒ `insufficient` — Atlas never fabricates.

## Durable storage — `atlas-storage/src/store.rs`

Embedded SurrealDB 1.5.6 (namespace/db `atlas`): the `mem` engine for
tests, **SurrealKV** at `<data_dir>/surreal` for durability. SurrealKV is
pure Rust; RocksDB was rejected because its C++ build requires libclang.
SurrealKV is gated upstream behind the `surrealdb_unstable` cfg, which is
set repo-locally in `.cargo/config.toml` — no machine-level RUSTFLAGS.
The lockfile pins surrealdb 1.5.6, so the "unstable" surface is fixed.
Tables are schemaless; the record structs in
`atlas-storage/src/records.rs` define shapes and IDs. Graph persistence
uses insertion ordinals and rebuilds through the graph's public builders,
re-firing M06's dedup/upgrade/evidence rules identically (this
reconstruction path found and fixed the `intern_predicate` bug).

## Hybrid retrieval — `atlas-storage/src/hybrid.rs`

Two stages, fused by reciprocal rank (RRF, constant 60): vector ranks come
from real cosine scores, graph ranks from PPR mass over the undirected
view of the relationship graph (damping 0.85, dangling mass teleports to
seeds). Ranks are real, so no fabricated similarity numbers enter the
ordering. Chunks found by both stages are marked `Both`; every result
carries per-stage provenance. Expansion is hop-bounded (default 1) and
cycle-safe; missing chunk records are skipped. Measured behavior is
two-sided (hybrid recovers graph-only evidence and can also displace
correct vector hits via tie-breaks) — see the M08 section of
[ROADMAP.md](ROADMAP.md).

## Extractive QA — `atlas-qa`

Answers quote learned sentences verbatim with citations and confidence;
the only "generation" is template assembly. Sentence ranking is
question-word overlap with fixed tie-breaks. Known limitations: overlap
scoring misses paraphrases; mean pooling (M05) is the only text-embedding
composition; multi-document evidence is flagged but contradictions are not
detected — they surface as adjacent verbatim quotes.

## Known limitations

- Per-query index rebuild (see Search above) — revisit if chunk counts
  reach ~10⁶ (see the analytical memory table in BENCHMARKS.md).
- No transactions: partial learn states are repaired by re-learning.
- Text normalization exists only in consumers (graph entity keys); the
  tokenizer stays lossless.
- Evaluation fixtures are deliberately small; numbers characterize the
  fixtures as much as the system.
