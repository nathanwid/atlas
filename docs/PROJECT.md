# Atlas — Project

> A Rust-native semantic learning engine that incrementally learns from text,
> builds embeddings and a knowledge graph, and answers questions using its own
> knowledge.

Atlas is an educational project that demonstrates the fundamental building
blocks behind modern AI systems. Rather than training a large language model,
Atlas focuses on learning semantic relationships from text and making that
knowledge searchable and explainable.

Atlas is designed to answer one question:

> **How can we build an intelligent knowledge engine from scratch?**

---

## Goals

### Primary goals

- Learn from natural language
- Build embeddings
- Construct a knowledge graph
- Perform semantic search
- Answer questions using stored knowledge
- Keep the architecture simple and understandable

### Non-goals

Atlas will **not** include:

- Transformer training
- Fine-tuning LLMs
- Multi-user authentication
- Web dashboards
- Distributed systems
- Plugin frameworks
- Production-scale infrastructure

External models (e.g. pretrained embedders or LLMs over HTTP) may be used as
optional integrations and baselines, but they are not the core of the project:
the point is that Atlas builds its own knowledge.

---

## Scope

Atlas is a **CLI-first** application. Everything should be usable from the
terminal.

```bash
atlas learn "Rust uses Tokio."

atlas similar rust

atlas graph rust

atlas search async

atlas ask "What does Rust use?"
```

---

## Core features

### 1. Tokenization

Convert text into tokens. Atlas uses a hand-rolled byte-level BPE tokenizer
(trained on text, verified against tiktoken's `cl100k_base`), so tokens are
byte-string pieces rather than plain words, and encoding is lossless.

Text normalization (lowercasing, stemming, etc.) is deliberately **not** part
of the tokenizer; it is a consumer-layer decision, deferred to the milestone
that first needs it (see `ROADMAP.md`).

### 2. Vocabulary

Assign every token a unique identifier and persist the mapping. Built on top
of the tokenizer's tokens (M03).

### 3. Embeddings

Represent chunks of learned text as dense vectors.

- M04: chunk documents and embed them through a pretrained baseline
  (FastEmbed / `BAAI/bge-small-en-v1.5`), with a content-addressed disk cache.
- M05: Atlas's **own** embeddings via Skip-Gram training and incremental
  learning. The pretrained path remains as a comparison baseline.

### 4. Knowledge graph

Store explicit relationships between entities.

```
Rust ──uses──▶ Tokio
```

### 5. Semantic search

Retrieve related documents, chunks, and entities using vector similarity,
later combined with graph traversal (hybrid retrieval).

### 6. Question answering

Answer questions using Atlas's own stored knowledge — knowledge graph plus
retrieved context — without requiring an LLM.

---

## Architecture

```
             User
               │
               ▼
          CLI Commands
               │
               ▼
          Tokenizer
               │
               ▼
          Vocabulary
               ▼
      Embedding Engine
               │
      ┌────────┴────────┐
      ▼                 ▼
Knowledge Graph   Vector Search
      │                 │
      └────────┬────────┘
               ▼
      Question Answering
```

---

## Tech stack

| Category       | Technology            | Purpose                              | Status |
| -------------- | --------------------- | ------------------------------------ | ------ |
| Language       | Rust                  | Primary programming language         | ✅      |
| CLI            | clap                  | Command-line interface               | ✅ (M01) |
| Async runtime  | tokio                 | Asynchronous runtime                 | 🔜      |
| Storage        | SurrealDB             | Documents, graph, and vector storage | 🔜 (M07) |
| Embeddings     | fastembed             | Pretrained embedding baseline (M04)  | ✅      |
| Serialization  | serde                 | Serialization framework              | ✅      |
| JSON           | serde_json            | JSON support                         | ✅      |
| Math           | ndarray               | Vector and matrix operations         | 🔜      |
| Error handling | anyhow                | Application errors                   | ✅      |
| Error handling | thiserror             | Library errors                       | ✅ (M01) |
| Logging        | tracing               | Structured logging                   | ✅ (M01) |
| Logging        | tracing-subscriber    | Log formatting and filtering         | ✅ (M01) |
| Configuration  | figment *(or config)* | Configuration management             | ✅ (M01) |
| Testing        | cargo test            | Unit and integration testing         | ✅      |
| Benchmarking   | criterion             | Performance benchmarks               | ✅      |
| Formatting     | rustfmt               | Code formatting                      | ✅      |
| Linting        | clippy                | Static analysis                      | ✅      |

Statuses reflect what is actually wired up today, not what is planned.

---

## Repository structure

```
atlas/
├── Cargo.toml
├── README.md
├── AGENTS.md
├── docs/
│   ├── PROJECT.md          ← this file
│   └── ROADMAP.md          ← milestones, status, sequencing
├── apps/
│   └── atlas-cli/          CLI entry point (M01)
├── crates/
│   ├── atlas-core/         shared config/error foundations (M01)
│   ├── atlas-tokenizer/    byte-level BPE (M02, complete)
│   ├── atlas-vocab/        token ID management + persistence (M03)
│   ├── atlas-embedding/    chunking + embedding pipeline (M04 groundwork)
│   ├── atlas-graph/        entity graph (M06)
│   └── atlas-storage/      index/search + persistence (M04/M07)
├── examples/
└── tests/
```

---

## Philosophy

- Simplicity over features
- Learning over scale
- Explainability over complexity
- Incremental development
- Clean architecture

The project should remain small enough for one developer to understand
completely while demonstrating the essential ideas behind semantic learning
systems.

---

## Stable project-level decisions

These are settled decisions. Changing any of them requires explicit approval.

1. **Atlas is a learning/knowledge engine, not a vector database.** Vector
   machinery exists to serve learning, search, and answering — not as an end
   in itself.
2. **No separate milestones for SIMD, Rayon, HNSW, or vector-search
   optimization.** Where performance work is needed, it is folded into the
   milestone that needs it.
3. **FastEmbed is a baseline, not the destination.** Atlas's own learned
   embeddings belong to M05 (Skip-Gram training and incremental learning),
   with the pretrained path kept as a comparison baseline.
4. **Storage, including SurrealDB/pgvector integration, belongs to M07.**
   Earlier index work is in-memory groundwork.
5. **Reuse existing infrastructure.** Do not duplicate what a crate already
   provides; do not unnecessarily rewrite working code.
6. **No text normalization inside the tokenizer.** The byte-level tokenizer
   stays lossless; normalization is a consumer-layer concern, deferred until a
   milestone needs it (decision recorded, not yet made).
7. **Vocabulary key format** (BPE token strings, byte sequences, or normalized
   words) is deferred to M03 planning.
8. **Vector-index code location** (currently `atlas-storage` groundwork on a
   branch) is revisited in M04 when similarity search is wired end to end.
9. **fastembed's dependency feature set is frozen** unless separately
   approved.
10. **New scope and new dependencies require approval first**, with a reason.
