# Atlas

Atlas is a Rust learning project: a semantic engine that incrementally learns
from text, builds embeddings and a knowledge graph, and answers questions
using its own knowledge — no LLM required.

The point is to understand how these systems work by implementing the
mechanisms by hand (tokenization, embeddings, graph retrieval) and measuring
against established baselines where honest numbers matter.

## Status

- **M01 Foundation** — complete: CLI wired (learn/similar/graph/search/ask),
  tracing logging, figment configuration.
- **M02 Tokenizer** — complete: hand-rolled byte-level BPE, cl100k-verified,
  trained + benchmarked.
- **M03 Vocabulary** — complete: byte-keyed dense token IDs, deterministic
  build, versioned binary persistence.
- **M04 Embedding Engine** — next; groundwork done: chunking, FastEmbed
  baseline, vector cache, and brute-force similarity with recall@k (pending
  merge from the `atlas/vector-search` branch).

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full milestone plan and
[docs/PROJECT.md](docs/PROJECT.md) for vision, architecture, and stable
project decisions.

## Build and test

```sh
cargo build
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo bench -p atlas-tokenizer
```
