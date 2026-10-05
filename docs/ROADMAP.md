# Atlas — Roadmap

Milestone sequence for the original Atlas vision. `docs/PROJECT.md` carries
the vision and stable decisions; this file carries sequencing, scope, and
current status.

## Status at a glance

| Milestone | Name               | Goal                                                    | Status |
| --------- | ------------------ | ------------------------------------------------------- | ------ |
| M01       | Foundation         | Workspace, CLI, logging, configuration, testing         | ✅ complete |
| M02       | Tokenizer          | Tokenization and text normalization                     | ✅ complete |
| M03       | Vocabulary         | Token ID management and persistence                     | ⏭ next |
| M04       | Embedding Engine   | Chunking, embeddings, similarity search                 | ◐ groundwork done |
| M05       | Learning Engine    | Skip-Gram training and incremental learning             | ⏳ pending |
| M06       | Knowledge Graph    | Entity extraction and relationship graph                | ⏳ pending |
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

## M03 — Vocabulary · ⏭ next

**Goal.** Token ID management and persistence.

**Planned scope.** `atlas-vocab` becomes real: a persistent token → ID
mapping built over the tokenizer's output, with load/save so learned
vocabulary survives runs.

**Dependencies.** M02 (tokens come from `atlas-tokenizer`).

**Open design decision (deferred to M03 planning).** Vocabulary key format:
BPE token strings, byte sequences, or normalized words. Normalization itself,
if adopted at this layer, is a consumer-side choice — the tokenizer stays
untouched.

---

## M04 — Embedding Engine · ◐ groundwork done

**Goal.** Chunking, embeddings, and similarity search over learned text.

**Groundwork already in place** (built ahead under the earlier milestone
numbering; to be merged from the `atlas/vector-search` branch):
- `atlas-embedding`: token-bounded document chunking at pre-token boundaries
  (with overlap + oversized-piece hard splits); `Embedder` trait with a
  FastEmbed backend (`BAAI/bge-small-en-v1.5`, 384-dim); content-addressed
  vector cache (SHA-256(model + text) → bincode).
- `atlas-storage` (on that branch): `VectorIndex` trait +
  `BruteForceIndex` — contiguous row-major storage, L2-normalized vectors,
  cosine-as-dot-product exact top-k — plus a committed query set with
  known-good chunks and recall@k evaluation (real-model eval is report-only).

**Remaining for M04.** Merge/land the groundwork branch; wire chunk → embed →
index → search end to end behind the CLI/learning flow; keep FastEmbed as the
pretrained baseline for M05 comparisons.

**Dependencies.** M02 (token counting), M03 (persistent vocabulary, if chunk
records need stable IDs), fastembed (approved).

---

## M05 — Learning Engine · ⏳ pending

**Goal.** Atlas's own embeddings: Skip-Gram training and incremental
learning. Atlas learns from text it is given and improves its embeddings over
time; the FastEmbed path remains as the comparison baseline.

**Dependencies.** M03 (vocabulary), M04 (chunk pipeline as training input).

---

## M06 — Knowledge Graph · ⏳ pending

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
| Token-bounded chunking (`atlas-embedding`) | M04 groundwork |
| FastEmbed embedder + `Embedder` trait | M04 — pretrained **baseline**; M05 adds Atlas's own learned embeddings |
| Vector cache (SHA-256 keyed, bincode) | M04/M07 — retained where useful; durability revisited in M07 |
| `BruteForceIndex` + `VectorIndex` + recall@k eval (branch `atlas/vector-search`) | M04 similarity-search groundwork, pending merge |
| `atlas-vocab` (empty) | M03 — deliberately unused until then |
| `atlas-cli` command placeholders (`learn/similar/graph/search/ask`) | M01 CLI wiring; commands activate milestone by milestone |

---

## Implementation notes

- Performance work (SIMD, Rayon, HNSW, scan optimization) has **no dedicated
  milestone**; it is folded into whichever milestone needs it.
- Storage/SurrealDB integration has **no foothold before M07**; earlier index
  work is intentionally in-memory.
- Repo-wide development conventions, commands, and gotchas live in
  `AGENTS.md`; current user-facing status lives in `README.md`.
