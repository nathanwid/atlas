# Atlas — Roadmap

Milestone sequence for the original Atlas vision. `docs/PROJECT.md` carries
the vision and stable decisions; this file carries sequencing, scope, and
current status.

## Status at a glance

| Milestone | Name               | Goal                                                    | Status |
| --------- | ------------------ | ------------------------------------------------------- | ------ |
| M01       | Foundation         | Workspace, CLI, logging, configuration, testing         | ✅ complete |
| M02       | Tokenizer          | Tokenization and text normalization                     | ✅ complete |
| M03       | Vocabulary         | Token ID management and persistence                     | ✅ complete |
| M04       | Embedding Engine   | Chunking, embeddings, similarity search                 | ✅ complete |
| M05       | Learning Engine    | Skip-Gram training and incremental learning             | ✅ complete |
| M06       | Knowledge Graph    | Entity extraction and relationship graph                | ⏭ next |
| M07       | Storage            | Persist documents, embeddings, and graph (SurrealDB)    | ⏳ pending |
| M08       | Semantic Search    | Hybrid vector and graph retrieval                       | ⏳ pending |
| M09       | Question Answering | Answer questions from Atlas knowledge                   | ⏳ pending |

---

## M01 — Foundation · ✅ complete

**Goal.** Workspace, CLI, logging, configuration, and testing foundation.

**Delivered.** Cargo workspace with test/bench/clippy/fmt conventions and
criterion. `atlas-cli` wired with clap: the five vision commands
(`learn`, `similar`, `graph`, `search`, `ask`) parse, log via `tracing`, and
dispatch — each reports the milestone that will deliver its functionality
(no stub logic). Logging via `tracing` + `tracing-subscriber`, configured by
`RUST_LOG` → `-v/-vv` → config. Configuration via `figment` in `atlas-core`
(defaults → optional `atlas.toml` → `ATLAS_*` env), with `thiserror`-based
`AtlasError`. Configuration is deliberately minimal (`log_level` only);
storage/data-path config is added by the milestone that needs it.

**Dependencies.** None.

---

## M02 — Tokenizer · ✅ complete

**Goal.** Tokenization and text normalization.

**Delivered.** Hand-rolled byte-level BPE in `atlas-tokenizer`: training on a
small corpus, encoding (two test-equivalent merge engines: naive rescan and
priority-queue + linked list), GPT-style pre-tokenization, tiktoken-format
rank parsing, exact encode/decode roundtrip. Token IDs are verified identical
to tiktoken-rs `cl100k_base` by diff tests; criterion benchmarks compare both
engines against tiktoken-rs and the HF `tokenizers` crate.

**Notes.**
- Tokens are byte-string pieces (lossless), **not** lowercased words. Text
  normalization (the original roadmap's "normalization") is a deferred
  consumer-layer decision and intentionally not in the tokenizer; folding it
  in would break cl100k parity and the exact roundtrip guarantee.
- `atlas-tokenizer::BpeTokenizer::token_byte_len` exists as a small
  chunking-serving extension.

---

## M03 — Vocabulary · ✅ complete

**Goal.** Token ID management and persistence.

**Delivered.** `atlas-vocab`: a `Vocab` keyed by raw token bytes with dense
`u32` IDs assigned by first occurrence, both-direction lookup, explicit
`Option`-based unknown handling (no reserved UNK), deterministic corpus
construction (`build`), incremental growth (`get_or_insert` — the M05 hook),
and versioned bincode persistence that treats corrupt files as errors, never
silently empty vocabularies. `atlas-tokenizer` gained one small reverse
lookup, `BpeTokenizer::token_bytes(rank)`; merge ranks stay in the tokenizer.

**Key decision (the deferred M03 question, now resolved).** Vocabulary keys
are raw token bytes: byte-level tokens can be invalid UTF-8, so bytes are the
only representation that preserves tokenizer semantics and exact
round-tripping. Rank-keyed vocabularies would be opaque without the merge
file; a normalized-word/term vocabulary is a different construct and remains
deferred until a milestone needs it (with the normalization decision that
would come with it).

**Dependencies.** M02 (done).

---

## M04 — Embedding Engine · ✅ complete

**Goal.** Chunking, embeddings, and similarity search over learned text.

**Delivered.** The groundwork (token-bounded chunking; `Embedder` trait with
a FastEmbed backend for `BAAI/bge-small-en-v1.5`, 384-dim; content-addressed
vector cache; `VectorIndex` trait + `BruteForceIndex` exact cosine top-k;
query set with known-good chunks and recall@k evaluation) was merged from
`atlas/vector-search` and completed with:

- **Query/document separation.** `Embedder::embed_documents` (batched, raw)
  vs `Embedder::embed_query` (single). BGE's query instruction prefix is
  isolated inside `FastembedEmbedder`; backends without query handling use
  the default pass-through. `embed_chunks` embeds documents only.
- End-to-end offline evaluation (deterministic fake embedder, ground truth
  by construction) and a report-only real-model evaluation (recall@1/3/5 +
  scan latency, no thresholds).

**Boundary notes.** Embeddings operate on document chunks; token-level
embeddings are M05's subject, so `atlas-vocab` is intentionally not consumed
yet. CLI `learn`/`search` stay wired-but-placeholder: making them real needs
chunk-text persistence, which belongs to M07 (SurrealDB). FastEmbed remains
the pretrained baseline; M05 adds Atlas's own learned embeddings alongside.

**Dependencies.** M02 (done), fastembed (approved).

---

## M05 — Learning Engine · ✅ complete

**Goal.** Atlas's own embeddings: Skip-Gram training and incremental
learning. Atlas learns from text it is given and improves its embeddings over
time; the FastEmbed path remains as the comparison baseline.

**Delivered.** `atlas-embedding::learn` — a hand-rolled Skip-Gram model with
negative sampling (no autodiff, no tensor library, single-threaded,
deterministic):

- **Training.** `train(&mut Vocab, &BpeTokenizer, corpus, &TrainerConfig)`:
  corpus → tokenizer → M03 vocabulary IDs (growing it via `get_or_insert` —
  one token-ID system), sliding-window pairs, unigram^0.75 negative sampling
  from a seeded xorshift PRNG (no `rand`), linear lr decay, SGD on
  center/context matrices.
- **Incremental learning.** `SkipGramModel::learn`: new text grows the
  vocabulary and appends initialized rows, then trains on the new text at
  the decay floor. Documented limitation: no replay buffer, so old vectors
  can drift.
- **Inference.** Token-level API (`token_vector`/`similarity`/`nearest`) plus
  `LearnedEmbedder`, which implements the `Embedder` trait via mean pooling
  of known-token vectors (unknown tokens skipped; all-unknown text → zero
  vector).
- **Persistence.** Versioned bincode; loading validates the model was
  trained against the provided vocabulary (row-count check).
- **Evaluation.** Deterministic bit-identical reruns (unit-tested); corpus
  training on 9.9 KB / vocab 798 / dim 64 finishes in ~0.5 s (release) and
  shows honest co-occurrence neighbors (e.g. `search` → `connected`,
  `Graph`, `alike`). Report-only; no quality thresholds.

**Dependencies.** M03 (vocabulary IDs), M04 (crate + baseline).

---

## M06 — Knowledge Graph · ⏭ next

**Goal.** Entity extraction and an explicit relationship graph
(`Rust ──uses──▶ Tokio`), plus graph-backed retrieval helpers.

**Dependencies.** M03 (entity identity via vocabulary), M04/M05 (semantics
for entity linking).

---

## M07 — Storage · ⏳ pending

**Goal.** Persistence: documents, embeddings, and graph stored in SurrealDB.
All in-memory index/cache work from M04 becomes durable here; pgvector-style
vector storage comparisons belong to this milestone too.

**Dependencies.** M04 (shapes to persist), M06 (graph to persist).

---

## M08 — Semantic Search · ⏳ pending

**Goal.** Hybrid vector and graph retrieval: combine similarity search with
graph traversal, evaluated honestly (recall/latency) against the single-signal
approaches.

**Dependencies.** M04 (vector search), M06 (graph).

---

## M09 — Question Answering · ⏳ pending

**Goal.** Answer questions from Atlas's own knowledge: retrieve relevant
facts via M08 and compose answers from stored relationships. No LLM required.

**Dependencies.** M08.

---

## Reconciliation map (old numbering → original roadmap)

Code that already exists was built under a temporary milestone numbering; the
original roadmap above is the source of truth going forward.

| Existing work | Home in the original roadmap |
| --- | --- |
| Byte-level BPE tokenizer (`atlas-tokenizer`) | M02 (complete) |
| Token-bounded chunking (`atlas-embedding`) | M04 (complete) |
| FastEmbed embedder + `Embedder` trait | M04 — pretrained **baseline**; M05 adds Atlas's own learned embeddings |
| Vector cache (SHA-256 keyed, bincode) | M04 (complete); durability revisited in M07 |
| `BruteForceIndex` + `VectorIndex` + recall@k eval | M04 (complete, merged from `atlas/vector-search`) |
| `atlas-vocab` | M03 (complete) — consumed by M05, not M04 |
| `atlas-cli` command placeholders (`learn/similar/graph/search/ask`) | M01 wired the commands; behavior arrives per milestone — `learn`/`search` need M07 persistence first |

---

## Implementation notes

- Performance work (SIMD, Rayon, HNSW, scan optimization) has **no dedicated
  milestone**; it is folded into whichever milestone needs it.
- Storage/SurrealDB integration has **no foothold before M07**; earlier index
  work is intentionally in-memory.
- Repo-wide development conventions, commands, and gotchas live in
  `AGENTS.md`; current user-facing status lives in `README.md`.
