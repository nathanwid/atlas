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
- **M04 Embedding Engine** — complete: chunking, FastEmbed baseline, vector
  cache, brute-force similarity with recall@k evaluation.
- **M05 Learning Engine** — complete: hand-rolled deterministic Skip-Gram
  with negative sampling, incremental learning, and mean-pooling text
  embeddings; FastEmbed stays as the baseline.
- **M06 Knowledge Graph** — complete: deterministic entity/relationship
  extraction, directed graph with evidence, BFS traversal; std-only.
- **M07 Storage** — next.

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
