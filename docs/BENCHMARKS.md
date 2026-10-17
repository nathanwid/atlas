# Atlas Benchmarks

All numbers on this page were produced by running the committed Criterion
benchmarks on the author's machine. They are real measurements, not
estimates; treat fine-grained differences as noise and order-of-magnitude
differences as signal. Re-run the commands below to reproduce on your
hardware.

## Environment

- OS: Windows 11 (x86_64), rustc 1.96 stable, MSVC toolchain
- Run from a warm `cargo bench` (compilation excluded from timings)
- Engines: SurrealDB in-memory engine (`mem`) for storage benches —
  durability I/O is deliberately excluded to isolate pipeline cost

## Commands

```sh
cargo bench -p atlas-tokenizer              # M02: merge engines vs tiktoken vs HF
cargo bench -p atlas-storage --bench hardening   # ingestion + search latency
cargo bench -p atlas-graph --bench pagerank      # personalized PageRank scaling
```

## Methodology

- **Fixtures are deterministic.** Documents are generated from a seeded
  xorshift PRNG over a 16-word vocabulary (30 words per document ≈ 30
  tokens ≈ one chunk). Fixture construction and store creation happen in
  Criterion setup and are **excluded** from timings.
- **Ingestion** benches time `atlas_storage::learn` for N *unique*
  documents (a global counter guarantees uniqueness so the duplicate
  short-circuit never fires). Reported as elements/s where one element is
  one document (chunk + fake-embedding + graph extraction + upserts).
- **Search** benches distinguish three cost tiers, labeled by name:
  - `query_only` — in-memory `BruteForceIndex` exact scan alone (µs).
  - `load_rebuild_query` — embedding-table full scan + index rebuild +
    the same scan.
  - `workflow_search` / `hybrid_search_full` — the complete M07/M08
    paths, including `embed_query`, chunk hydration, and (hybrid) graph
    load + personalized PageRank + RRF fusion.
- **Fake embedder.** Storage benches use a deterministic bag-of-words
  embedder (512 dims) instead of FastEmbed, isolating pipeline cost from
  model inference. Real-model retrieval quality numbers live in the M08
  evaluation tests, not here.
- **Sample sizes** are reduced (10–20) for the expensive suites to keep
  total run time reasonable; Criterion still reports confidence intervals.
- **Execution**: these benches have `harness = false` and run only via
  `cargo bench`; `cargo test` compiles them but never executes them.

## Results (measured 2026-10, rustc 1.96, Windows 11)

### Personalized PageRank (atlas-graph)

| graph | edges | median time |
|---|---|---|
| ring + chords, 20 PPR iterations, 2 seeds | ~1,000 | 1.55 ms |
| ring + chords, 20 PPR iterations, 2 seeds | ~10,000 | 132.5 ms |

Scaling is ~10× time for ~10× edges — linear, as expected for a fixed
iteration count.

### Ingestion (atlas-storage, mem engine, fake embedder)

| documents | batch time | throughput |
|---|---|---|
| 10 | 10.9 ms | ~921 docs/s |
| 100 | 123.6 ms | ~809 docs/s |
| 1,000 | 1.264 s | ~791 docs/s |

Throughput is roughly flat from 100 to 1,000 documents (~1.26 ms per
document end to end: chunking + fake embedding + graph extraction +
database upserts).

### Search latency (atlas-storage, mem engine, fake embedder)

| stage | 100 chunks | 1,000 chunks |
|---|---|---|
| `query_only` (in-memory exact scan) | 62 µs | 625 µs |
| `load_rebuild_query` (+ table load + rebuild) | 25.4 ms | 237.4 ms |
| `workflow_search` (full M07 path) | 22.3 ms | 203.2 ms |
| `hybrid_search_full` (full M08 path) | 22.8 ms | 203.6 ms |

Readings:

- The exact scan itself is negligible (62–625 µs) at Atlas scale; the
  dominant cost of every durable search path is **reloading the embedding
  table from SurrealDB and rebuilding the index per query** — by design,
  since M07 chose simplicity over an index cache.
- Hybrid adds no measurable cost over vector-only at this scale (the
  fixture graph is small; PPR is microseconds to low milliseconds).
- 10× chunks scales `query_only` ~10× (linear O(n·d), as documented) and
  load+rebuild ~9–10× (linear in rows scanned).

## Analytical memory estimates (NOT measured RSS)

These are arithmetic estimates from the data model, not measurements. They
cover **embedding-vector storage only** — the dominant term for search —
and omit: hash-map overhead (~24–48 B/entry plus allocator slack), chunk
text strings, SurrealDB's own page caches, the OS page cache, and process
baseline.

| component | formula | example (1,000 chunks, 384-dim FastEmbed) |
|---|---|---|
| `embedding` rows (DB) | chunks × dim × 4 B | 1,000 × 384 × 4 ≈ **1.5 MiB** |
| `BruteForceIndex` (in-memory copy) | chunks × dim × 4 B | ≈ **1.5 MiB** |
| query-time peak (index + vectors snapshot) | 2 × above | ≈ **3 MiB** |
| M05 Skip-Gram model file | 2 × vocab × dim × 4 B | 798 × 64 × 2 × 4 B ≈ **0.4 MiB** |

Conclusion: at Atlas's current scale, vector memory is a few MiB — no
allocator instrumentation is warranted yet. If chunk counts reach ~10⁶,
the same formula gives ~1.5 GiB for the in-memory index, at which point
the load-into-BruteForceIndex decision should be revisited.

## Limitations

- Storage benches use the in-memory engine; durable (SurrealKV) write
  costs are **not** represented in the ingestion numbers.
- The fake embedder removes model inference; real FastEmbed ingestion is
  model-inference-dominated and was measured during M07 (≈ tens of ms per
  small batch, plus a one-time ~100 MB model download).
- Windows/Defender adds run-to-run noise to whole-process timings; the
  tables above are single `cargo bench` sessions and should be compared
  at order-of-magnitude granularity. A verified repeat run of
  `ingestion/100_docs` landed within noise (+1.0%, p = 0.55) while
  `query_only/1000_chunks` shifted +11% — same conclusions, different
  decimals.
- Only the tokenizer, storage, and graph crates have benchmarks; the QA
  layer is a thin composition of benchmarked paths and was not measured
  separately.
