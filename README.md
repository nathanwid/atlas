# Atlas

Atlas is a Rust learning project: a semantic engine that incrementally learns
from text, builds embeddings and a knowledge graph, and answers questions
using its own knowledge — no LLM required.

The point is to understand how these systems work by implementing the
mechanisms by hand (tokenization, embeddings, graph retrieval) and measuring
against established baselines where honest numbers matter.

## Status

All nine roadmap milestones are complete — tokenizer, vocabulary,
embedding engine, learning engine, knowledge graph, storage, semantic
search, and question answering. See
[docs/ROADMAP.md](docs/ROADMAP.md) for what shipped in each,
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit
together, [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for measured results,
and [docs/PROJECT.md](docs/PROJECT.md) for vision and stable project
decisions.

## Quickstart

Prerequisites: Rust 1.89 or newer (stable). `learn`, `search`, and `similar`
can download the FastEmbed model (~100 MB) into `.fastembed_cache/` on first
use. Later runs are offline. Atlas creates `atlas-data/` as needed. Git
ignores this directory.

```sh
cargo run -p atlas-cli -- learn "Rust uses Tokio for async runtimes."
# learn: stored 35 (id 85e6be14…) — 1 chunks, 1 embeddings, 2 entities, 1 relationships.

cargo run -p atlas-cli -- graph rust
# graph: Rust (within 1 hop)
#   Rust -[uses]-> Tokio (explicit, outgoing)
#     evidence: "Rust uses Tokio for async runtimes" [doc 85e6be14…, chunk 85e6be14…:0]

cargo run -p atlas-cli -- similar rust -k 5
# similar: labels closest to "rust"
# #0 Tokio cosine=<score>

cargo run -p atlas-cli -- search "tokio async runtime" -k 3
# #0 source=vector score=0.8204 chunk=85e6be14… doc=85e6be14…
#   via vector rank 1 (score 0.8204)
#   Rust uses Tokio for async runtimes.

cargo run -p atlas-cli -- ask "What uses Tokio?"
# From the knowledge graph: Rust -[uses]-> Tokio. [doc 85e6be14…, chunk 85e6be14…:0]
# From the learned content:
# 1. "Rust uses Tokio for async runtimes" [doc 85e6be14, chunk 85e6be14]
# Confidence: high.
```

Learning complete text twice with the same model is a no-op. Learning with
another model adds its embeddings; missing derived records are repaired.
FastEmbed rejects inputs above its actual token limit. Use shorter passages
and learn them separately. `similar` compares entity labels, not entity
contexts. It ranks labels with cosine scores and accepts `-k/--top` (default
5). `graph` matches trimmed, case-insensitive entity names. It accepts
`--hops` from 0 to 3 (default 1) and `--limit` (default 50 relationship
rows). It checks evidence against live chunks before it shows text or cites
it. Empty or whitespace-only `learn` input is rejected as a usage error.
Override the data directory with `ATLAS_DATA_DIR=/some/dir` or an
`atlas.toml` file.

## Build and test

```sh
cargo build
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Benchmarks (tokenizer, ingestion, search latency, PageRank scaling) are
measured and documented in [docs/BENCHMARKS.md](docs/BENCHMARKS.md):

```sh
cargo bench -p atlas-tokenizer
cargo bench -p atlas-storage --bench hardening
cargo bench -p atlas-graph --bench pagerank
```

## Troubleshooting

Storage is an embedded SQLite database at `atlas-data/atlas.db` (WAL
journal, `synchronous=FULL`). Raise CLI log verbosity when debugging:

```sh
cargo run -p atlas-cli -vv -- search "tokio"
```
