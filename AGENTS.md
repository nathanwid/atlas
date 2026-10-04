# AGENTS.md

## What this repo is

Atlas is a Rust learning project for understanding how semantic search and
knowledge graph systems work. It builds a search engine milestone by
milestone, implementing mechanisms by hand and measuring them against
established libraries. `README.md` reflects the current milestone state.

Layout: `apps/atlas-cli` + `crates/atlas-{core,vocab,tokenizer,embedding,graph,storage}`.
Most crates are deliberate empty placeholders for future milestones — do not
delete them or fill them ahead of their milestone. `atlas-vocab` is unused on
purpose until a concrete term-vocabulary need exists.

## Milestone roadmap (high-level)

1. BPE tokenizer
2. Document chunking and embeddings
3. Brute-force vector search and evaluation
4. Vector search optimization
5. HNSW
6. Entity graph and PageRank
7. Hybrid vector + graph retrieval
8. Service layer and alternative vector-storage backends

## Workflow rules

- Inspect and reuse the existing project structure before making changes.
- Work on one milestone at a time. Do not implement future milestones unless
  explicitly requested.
- Before implementing a milestone or significant feature: inspect the project,
  present a brief plan with acceptance criteria, and wait for explicit
  approval. Small fixes, formatting, and straightforward corrections do not
  need a separate planning step unless they materially change the design.
- Keep code idiomatic, simple, readable, and easy to understand; prefer clear
  implementations over clever abstractions. Explain important non-obvious
  implementation decisions.
- Do not add dependencies without asking first and explaining why.
- Add tests for new behavior where practical.
- Run the appropriate tests, Clippy, formatting checks, and benchmarks when
  applicable. Keep benchmark results honest — never invent measurements, and
  report losses to established libraries honestly.
- Never commit or push unless explicitly asked.

## Git conventions

- `main` represents the latest stable Atlas state.
- One branch per milestone, named `atlas/<milestone>` (e.g. `atlas/m1-bpe`).
- Do not commit directly to `main` while working on a milestone.
- Do not automatically create or switch branches; branch management is
  handled by the developer.

## Commands

- Full milestone gate (all three must pass):
  `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`
- One package: `cargo test -p atlas-tokenizer`
- One test: `cargo test -p atlas-tokenizer -- atlas_matches_tiktoken_on_samples`
- Benches (criterion, `harness = false` is configured in Cargo.toml):
  `cargo bench -p atlas-tokenizer`; filter a group: `cargo bench -p atlas-tokenizer -- long_piece`

## Tokenizer gotchas (milestone 1, `atlas-tokenizer`)

- `CL100K_PATTERN` in `src/pretokenize.rs` must stay byte-identical to the
  cl100k pattern in tiktoken-rs 0.12 (the diff tests depend on it). It uses
  lookahead + possessive quantifiers, so it needs `fancy-regex`, not `regex`.
- The `tokenizers` dev-dependency is pinned to its `fancy-regex` backend
  (`default-features = false`). Its default `onig` backend mis-parses
  `\p{N}{1,3}+` as an unbounded repeat and will break
  `hf_tokenizer_derived_from_ranks_matches_oracle`.
- Fixtures: `tests/fixtures/cl100k.tiktoken` (1.68 MB committed data) is only
  for the diff test against tiktoken-rs. `tests/fixtures/corpus.txt` is only
  for Atlas's own trainer. Never cross-feed them.
- Both merge engines (`encode_piece_naive`, `encode_piece_heap`) must produce
  identical output; equivalence is test-enforced. Keep tie-breaking
  deterministic (no HashMap-iteration-order dependence).
- `tests/common/mod.rs` is shared between integration tests and the bench via
  `#[path]`; its module-wide `#![allow(dead_code)]` is intentional because
  each test binary uses a different subset.

## Embedding gotchas (milestone 2, `atlas-embedding`)

- fastembed depends on `tokenizers` with the `onig` feature, and cargo feature
  unification means onig wins in any build graph containing fastembed. Never
  feed the cl100k pattern to HF `tokenizers`' own Split; drive it piecewise
  through `common::hf_encode_piecewise` (Atlas pre-tokenizes, HF only maps
  bytes + merges). The HF parity test enforces this.
- Keep `esaxx_fast` OFF the `tokenizers` features (see root Cargo.toml): its
  C++ backend builds with a static CRT that fails to link against ort's
  prebuilt onnxruntime on Windows (LNK2038) in debug builds.
- The real-model test is `#[ignore]`d (downloads ~100 MB of ONNX weights on
  first run, needs network). Run it explicitly:
  `cargo test -p atlas-embedding --release -- --ignored`.
- fastembed writes its model-weight cache to `.fastembed_cache/` inside the
  crate dir when tests run there; it is gitignored, never commit it.
- The chunker's hard-split of oversized pre-tokens cuts at token boundaries
  that are also char boundaries; such windows may exceed `max_tokens` when a
  token spans a character (see `chunker.rs`).

## Conventions

- `rustfmt.toml`: Unix newlines, edition 2024 — preserve.
