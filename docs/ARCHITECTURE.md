# Atlas Architecture

This document describes the system as implemented on `main` — every claim
is traceable to a source file listed inline. For milestone history see
[ROADMAP.md](ROADMAP.md); for measured numbers see
[BENCHMARKS.md](BENCHMARKS.md).

## Crate map

| crate | responsibility | depends on |
|---|---|---|
| `atlas-cli` (`apps/`) | clap CLI: learning, graph neighborhoods, entity-label similarity, search, and QA | core, embedding, qa, storage, tokenizer |
| `atlas-core` | figment configuration (`Config`), `AtlasError` | figment, serde, thiserror |
| `atlas-tokenizer` | byte-level BPE: pre-tokenization, ranks parsing, training, encoding (naive + heap engines), decoding | fancy-regex |
| `atlas-vocab` | token → dense stable IDs, deterministic build, versioned bincode persistence | tokenizer |
| `atlas-embedding` | chunking, `Embedder` trait + FastEmbed backend, Skip-Gram learning, vector cache | tokenizer, vocab, fastembed |
| `atlas-graph` | entities, directed relationships with evidence, extraction, BFS, personalized PageRank — **std-only, zero Atlas deps** | — |
| `atlas-storage` | `VectorIndex`/`BruteForceIndex`, SQLite store (`SqliteStore`), learn/search workflows, hybrid retrieval | embedding, graph, tokenizer, rusqlite, sha2 |
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
     → one batched FastEmbed call → document/chunk/embedding/graph records
     → per-chunk graph extraction (evidence doc_id = chunk_id)
     → one grouped write (Backend::apply_learn_write)
```

All records are idempotent upserts keyed by deterministic IDs. SQLite
commits everything one `learn` call produced in a **single transaction**
— all-or-nothing per document. Older partial states (a pre-batching
database, or rows deleted out from under a document) are still repaired
by the next `learn`: duplicate detection checks expected chunk text,
requested-model embeddings, and graph records/evidence by key. Missing
derived records trigger an atomic replay. A second model can index the
same document.

Each document supplies a graph delta. Within the same write transaction,
relationship evidence is merged by `(chunk_id, sentence)`, in first-seen
order. Existing labels, kinds, insertion order, and stronger origins are
preserved. New graph records append; graph loading restores kinds and
reads all tables within one SQLite snapshot.
This prevents further evidence loss; it does not restore evidence deleted
by older versions. Such recovery requires graph re-extraction from retained
chunks or corpus replay. Direct relationship upserts still replace evidence.

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

## Durable storage — `atlas-storage/src/sqlite_store.rs`

Embedded SQLite via `rusqlite` (bundled SQLite ≥ 3.51.3, asserted at
runtime by a version-guard test) at `<data_dir>/atlas.db`: WAL journal,
`synchronous=FULL`, 5 s busy timeout. Seven relational tables
(`document`, `chunk`, `embedding`, `entity`, `predicate`,
`relationship`, `evidence`) mirror the record shapes in
`atlas-storage/src/records.rs`; `learn` runs as one atomic transaction
through `Backend::apply_learn_write`. A bundled engine was chosen over
the previous SurrealDB 1.5.6/SurrealKV stack after the storage
evaluation measured equal-or-better read latency, specified durability
semantics, no unstable cfg gates, and a far smaller dependency surface;
the SurrealKV data directory `atlas-data/surreal/` survives only as a
forensic artifact. Graph persistence uses insertion ordinals and
rebuilds through the graph's public builders, re-firing M06's
dedup/upgrade/evidence rules identically (this reconstruction path
found and fixed the `intern_predicate` bug).

## Hybrid retrieval — `atlas-storage/src/hybrid.rs`

Two stages, fused by reciprocal rank (RRF, constant 60): vector ranks come
from real cosine scores, graph ranks from PPR mass over the undirected
view of the relationship graph (damping 0.85, dangling mass teleports to
seeds). Ranks are real, so no fabricated similarity numbers enter the
ordering. Chunks found by both stages are marked `Both`; every result
carries per-stage provenance. Shared candidates receive both rank
contributions and can change vector order. Graph-only mode excludes vector
seeds before ranking. Expansion is hop-bounded (default 1) and cycle-safe;
missing chunk records are skipped; vector search continues to fill live result slots. Fusion can change recall in either
direction; graph-only recovery in the final top-k is not guaranteed.

## Extractive QA — `atlas-qa`

Answers quote learned sentences verbatim with citations and confidence;
the only "generation" is template assembly. Sentence ranking uses unique content-word overlap, with punctuation
separated and a fixed function-word list. Confidence uses the selected
sentence vector score. Citations retain full chunk IDs. Known
limitations: lexical overlap
scoring misses paraphrases; mean pooling (M05) is the only text-embedding
composition; multi-document evidence is flagged but contradictions are not
detected — they surface as adjacent verbatim quotes.

Graph lead-ins match complete entity phrases. Each claim cites the first
live supporting chunk that contains its evidence sentence. Validation
reuses retrieved text and caches other chunk lookups, including misses.

## Known limitations

- Per-query index rebuild (see Search above) — revisit if chunk counts
  reach ~10⁶ (see the analytical memory table in BENCHMARKS.md).
- One durable write transaction per `learn` — durability batching across
  documents is deliberately not done (it would weaken per-document
  atomicity); per-transaction fsync cost is visible in durable
  ingestion benchmarks.
- Text normalization exists only in consumers (graph entity keys); the
  tokenizer stays lossless.
- Evaluation fixtures are deliberately small; numbers characterize the
  fixtures as much as the system.
- The transitive `paste` proc macro has advisory RUSTSEC-2024-0436 because it
  is unmaintained. The advisory reports no vulnerability, but `cargo audit
  --deny warnings` fails. FastEmbed still depends on Tokenizers 0.23, which
  uses `paste`; follow up when upstream provides a compatible replacement.

## Input and checkpoint compatibility

The fixed chunk layout remains `{doc_id}:{ordinal}`. FastEmbed validates
actual model-token counts before embedding, duplicate detection, or cache
reuse. Counts include special tokens and the BGE query prefix. Oversized
inputs return an error with instructions to shorten or split the text;
Atlas does not silently truncate or automatically change stored chunk IDs.
Grouped learning rejects different text or metadata for an existing chunk ID,
so older model vectors keep their original meaning. Keep the original
tokenizer/chunk settings for repair; use a new data directory for a changed
layout. Low-level chunk upserts retain their explicit replacement contract.

Skip-Gram model format v2 stores ordered-vocabulary and tokenizer
fingerprints. Inference IDs also include weights and pooling version, so
old learned vectors/cache entries cannot be reused under new model IDs.
The vocabulary file format remains v1. Legacy model v1 files cannot prove
their vocabulary mapping: loading rejects them and saving refuses to replace
them. Keep those files; retrain from the original corpus into new model and vocabulary
paths, then re-index with the new model identity. No automatic migration or
deletion occurs.

Model and vocabulary saves sync a temporary file before atomic replacement.
Separate saves are not one transaction; fingerprint checks reject mismatched
pairs. A paired-checkpoint publication API remains optional future work.
Process-kill tests cover open SQLite connections and unfinished grouped
transactions, including an evidence merge. They do not simulate power loss.
