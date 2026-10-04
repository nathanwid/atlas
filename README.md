# Atlas

Atlas is a Rust learning project that builds a semantic search and knowledge
graph engine piece by piece.

The goal is to understand how the pieces work by implementing the important
mechanisms by hand — tokenization, embeddings, vector indexes, and graph
retrieval — and by measuring Atlas's own implementations against established
libraries instead of assuming they are slower.

Main areas:

- **Tokenization** — a hand-rolled byte-level BPE tokenizer (training,
  encoding, merging)
- **Embeddings** — document chunk embedding with pluggable embedder backends
- **Vector search** — brute-force and approximate nearest-neighbor indexes,
  with recall@k and latency evaluation
- **Knowledge graphs** — entity extraction, graph expansion, personalized
  PageRank, and hybrid retrieval

## Status

Milestone 1 is complete: `crates/atlas-tokenizer` contains a byte-level BPE
tokenizer with training, a priority-queue and a naive merge engine, GPT-style
pre-tokenization, and benchmarks. It produces token IDs identical to
tiktoken-rs `cl100k_base`, verified by diff tests. The other workspace crates
are placeholders for future milestones.

## Build and test

```sh
cargo build
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo bench -p atlas-tokenizer
```
