# AGENTS.md

## What this repo is

Atlas is a Rust-native semantic **learning engine**: it incrementally learns
from text, builds embeddings and a knowledge graph, and answers questions from
its own knowledge — no LLM required. It is a learning project: mechanisms are
implemented by hand and measured against established baselines. It is a
knowledge engine, not a vector database.

Authoritative docs:

- `docs/PROJECT.md` — vision, goals/non-goals, architecture, tech stack, and
  stable project-level decisions.
- `docs/ROADMAP.md` — milestone scope, sequencing, and current status.
- `README.md` — user-facing status snapshot.

Layout: `apps/atlas-cli` + `crates/atlas-{core,vocab,tokenizer,embedding,graph,storage}`.
Most crates are deliberate empty placeholders for future milestones — do not
delete them or fill them ahead of their milestone. `atlas-vocab` owns token
ID management and persistence (M03); BPE mechanics stay in
`atlas-tokenizer`.

## Milestones

The milestone list is M01–M09 (Foundation, Tokenizer, Vocabulary, Embedding
Engine, Learning Engine, Knowledge Graph, Storage, Semantic Search, Question
Answering). Scope, deliverables, and status per milestone live in
`docs/ROADMAP.md` — do not rely on this file for sequencing details.

Current status: M01–M04 complete, **M05 Learning Engine is next** (Skip-Gram
over M03 vocabulary IDs; FastEmbed stays as the comparison baseline).

## Workflow rules

- Inspect and reuse the existing project structure before making changes.
  Do not unnecessarily rewrite working code.
- Work on one milestone at a time. Do not implement future milestones early;
  M03 (Vocabulary) is the next one — do not start it without an approved
  plan, and never start later milestones unprompted.
- Before implementing a milestone or significant feature: inspect the project,
  present a brief plan with acceptance criteria, and wait for explicit
  approval. Small fixes, formatting, and straightforward corrections do not
  need a separate planning step unless they materially change the design.
- Keep code idiomatic, simple, readable, and easy to understand; prefer clear
  implementations over clever abstractions. Explain important non-obvious
  implementation decisions.
- Do not add dependencies or expand scope without asking first and explaining
  why. Stable architectural decisions are listed in `docs/PROJECT.md`; they
  change only with explicit approval.
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
