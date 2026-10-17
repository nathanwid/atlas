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

Prerequisites: Rust 1.85 or newer (stable). The first `learn` downloads the
FastEmbed model (~100 MB) into `.fastembed_cache/` — later runs are
offline. All data lives under `atlas-data/` (created on demand;
gitignored).

```sh
cargo run -p atlas-cli -- learn "Rust uses Tokio for async runtimes."
# learn: stored 35 (id 85e6be14…) — 1 chunks, 1 embeddings, 2 entities, 1 relationships.

cargo run -p atlas-cli -- search "tokio async runtime" -k 3
# #0 source=vector score=0.8204 chunk=85e6be14… doc=85e6be14…
#   via vector rank 1 (score 0.8204)
#   Rust uses Tokio for async runtimes.

cargo run -p atlas-cli -- ask "What uses Tokio?"
# From the knowledge graph: Rust -[uses]-> Tokio.
# From the learned content:
# 1. "Rust uses Tokio for async runtimes" [doc 85e6be14, chunk 85e6be14]
# Confidence: high.
```

Learning the same text twice is a no-op (content-addressed documents).
Empty or whitespace-only `learn` input is rejected as a usage error.
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

SurrealDB SDK logging is filtered out by default (its shutdown path emits
harmless ERROR noise). To see engine-level logs when debugging storage:

```sh
RUST_LOG=surrealdb=info cargo run -p atlas-cli -- search "tokio"
```
