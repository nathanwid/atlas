# Atlas Benchmarks

All numbers on this page were produced by running the committed Criterion
benchmarks on the author's machine. They are real measurements, not
estimates; treat fine-grained differences as noise and order-of-magnitude
differences as signal. Re-run the commands below to reproduce on your
hardware.

## Environment

- OS: Windows 11 (x86_64), MSVC toolchain (rustc 1.96 for the 2026-10
  tables below; 1.99.1 re-verifies Clippy in CI)
- Run from a warm `cargo bench` (compilation excluded from timings)
- Engines: storage benches run on SQLite. `hardening` uses in-memory
  SQLite (`sqlite_mem`) — durability I/O excluded to isolate pipeline
  cost; `sqlite_durability` additionally measures durable SQLite
  (`sqlite_durable`: WAL + `synchronous=FULL`, one fsync per `learn`).
  The historical SurrealDB-mem and SurrealKV-durable tables are kept
  below, clearly marked: SurrealKV commits with default `Eventual`
  durability (no per-commit fsync), so its durable numbers are **not**
  comparable to SQLite's `synchronous=FULL`.

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

These historical timings increase about 85 times for 10 times the edges.
The old adjacency setup scanned all relationships once per entity (O(V*E));
the numbers do not show linear scaling. See the P2 measurements below.

### Ingestion (atlas-storage, storage engine, fake embedder)

Historical — SurrealDB in-memory engine (`mem`), before the SQLite
migration:

| documents | batch time | throughput |
|---|---|---|
| 10 | 10.9 ms | ~921 docs/s |
| 100 | 123.6 ms | ~809 docs/s |
| 1,000 | 1.264 s | ~791 docs/s |

Throughput is roughly flat from 100 to 1,000 documents (~1.26 ms per
document end to end: chunking + fake embedding + graph extraction +
database upserts).

The `hardening` bench now runs the same groups on in-memory SQLite;
re-run it for current numbers (not comparable to the SurrealDB table —
different engine). For durable SQLite ingestion, the PoC's
`sqlite_durability` bench measured 100 one-chunk documents at
~345–475 ms/100 docs (~210–290 docs/s) with `synchronous=FULL`; the
A/B experiment that introduced per-`learn` transaction batching
measured sequential autocommits at 1011 ms vs 373 ms grouped (median
of 10 paired runs) on the same machine.

### Search latency (atlas-storage, mem engine, fake embedder)

Historical — SurrealDB in-memory engine, before the SQLite migration:

| stage | 100 chunks | 1,000 chunks |
|---|---|---|
| `query_only` (in-memory exact scan) | 62 µs | 625 µs |
| `load_rebuild_query` (+ table load + rebuild) | 25.4 ms | 237.4 ms |
| `workflow_search` (full M07 path) | 22.3 ms | 203.2 ms |
| `hybrid_search_full` (full M08 path) | 22.8 ms | 203.6 ms |

Under SQLite the same read paths measured 397–536 µs end to end at 100
chunks (`sqlite_durability`, durable engine) — the SurrealDB numbers
above were dominated by loading rows out of SurrealDB, which SQLite
does roughly 70× faster at this scale. Re-run the benches for current
exact numbers.

Readings:

- The exact scan itself is negligible (62–625 µs) at Atlas scale; under
  SurrealDB the dominant read cost was **reloading the embedding table
  and rebuilding the index per query** — by design, since M07 chose
  simplicity over an index cache.
- The historical hybrid fixture produced no graph. Those timings measure
  vector fallback and cannot establish traversal or PageRank overhead.
- 10× chunks scales `query_only` ~10× (linear O(n·d), as documented) and
  load+rebuild ~9–10× (linear in rows scanned).

## Analytical memory estimates (NOT measured RSS)

These are arithmetic estimates from the data model, not measurements. They
cover **embedding-vector storage only** — the dominant term for search —
and omit: hash-map overhead (~24–48 B/entry plus allocator slack), chunk
text strings, SQLite's page cache, the OS page cache, and process
baseline.

| component | formula | example (1,000 chunks, 384-dim FastEmbed) |
|---|---|---|
| `embedding` vector payload (SQLite; excludes row overhead) | chunks × dim × 4 B | 1,000 × 384 × 4 ≈ **1.5 MiB** |
| loaded embedding records (search snapshot) | chunks × dim × 4 B | ≈ **1.5 MiB** |
| `BruteForceIndex` (in-memory copy) | chunks × dim × 4 B | ≈ **1.5 MiB** |
| query-time vector payload (records + index) | 2 × above | ≈ **3 MiB** |
| M05 Skip-Gram model file | 2 × vocab × dim × 4 B | 798 × 64 × 2 × 4 B ≈ **0.4 MiB** |

Conclusion: at Atlas's current scale, vector memory is a few MiB — no
allocator instrumentation is warranted yet. If chunk counts reach ~10⁶,
the same formula gives ~1.5 GiB for the in-memory index, at which point
the load-into-BruteForceIndex decision should be revisited.

Search adds each loaded record's vector directly to the index. It does not
create an extra full-size vector batch while it builds the index.

## Limitations

- The SurrealDB-era tables above are historical (engine removed from the
  tree); they are kept to document the measured migration rationale.
  SurrealKV-durable ingestion was measured once during the storage
  evaluation (~554–750 docs/s for 100 docs) under its default `Eventual`
  commit durability, which does not fsync per commit — not comparable to
  SQLite's `synchronous=FULL`.
- The `hardening` bench's in-memory SQLite numbers will differ from the
  historical SurrealDB-mem table (different engine); re-run before
  quoting.
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

## P2 recheck (2026-11-08)

Windows/MSVC, rustc 1.96.0, optimized Criterion builds. Both runs use the
same fixtures. Warm-up and target measurement time were one second each;
PageRank uses 10 samples, hybrid search uses 20. The table gives Criterion
point estimates, not median or RSS measurements. Short runs show local
behavior and do not establish results for other hardware or corpora.

| benchmark | before adjacency index | after adjacency index |
|---|---:|---:|
| PageRank, about 1,000 edges | 1.139 ms | 0.169 ms |
| PageRank, about 10,000 edges | 106.92 ms | 1.797 ms |
| connected hybrid, 100 chunks | 1.027 ms | 1.031 ms |
| connected hybrid, 1,000 chunks | 9.676 ms | 8.565 ms |

Criterion detected lower PageRank times at both sizes and an approximately
11.5% reduction for 1,000-chunk hybrid retrieval. It detected no significant
change for 100-chunk hybrid retrieval. Incident-edge lists use additional
O(V+E) memory and preserve edge order. PageRank adjacency setup now visits
incident edges, rather than scanning every edge for every entity.

The corrected hybrid fixture is a connected entity ring backed by learned
chunks, with the same deterministic fake embedder. Setup asserts graph
edges and graph-derived result provenance; Both results are valid after P1.
It measures graph loading, expansion, PageRank, and fusion. The historical
empty-graph hybrid numbers above are not comparable to this fixture.

Commands used before and after the change (replace `--save-baseline` with
`--baseline` for the second run):

```sh
cargo bench -p atlas-graph --bench pagerank -- --save-baseline p2-before --warm-up-time 1 --measurement-time 1 --sample-size 10
cargo bench -p atlas-storage --bench hardening -- search/hybrid --save-baseline p2-before --warm-up-time 1 --measurement-time 1 --sample-size 10
```

### Final full suite and fixture control

Search setup now uses fixed local seeds, independent of the number of timed
ingestion iterations. A runtime guard checks reproducibility. The seed
sequence preserves the corpus used by the focused before/after hybrid table.
Ingestion still uses a separate counter so its timed documents remain unique.

The final in-memory SQLite run (fake embedder, final chunk-identity guard)
measured ingestion batches of 10/100/1,000 documents at 1.354/13.691/142.83 ms.
These are point estimates, not durable SQLite or real-model throughput.

Earlier repeated full-suite comparisons reported slower times for 10-document
ingestion (+6.1%), 1,000-chunk load/rebuild (+4.5%), and hybrid retrieval
(+7.4%), alongside improvements in other cases. The search corpus in those
runs depended on timed ingestion counts, so those search comparisons were
not controlled. No isolated ingestion cost estimate exists for the extra
chunk-identity check. Those mixed observations do not establish a code-only
improvement or regression.

With fixed search seeds, the final short full suite measured 1,000-chunk
hybrid retrieval at 11.283 ms (interval 10.033-13.014 ms, four outliers among
20 samples). One longer focused repeat, using a three-second warm-up and a
five-second target measurement, measured 8.562 ms (8.499-8.634 ms). At 100
chunks it measured 0.885 ms. This variance limits end-to-end performance
claims. The deterministic regression also verifies reduced edge inspections;
it does not guarantee latency. Keep both full and focused observations when comparing future work.

```sh
cargo bench -p atlas-storage --bench hardening -- --save-baseline p2-stable-final --warm-up-time 1 --measurement-time 1 --sample-size 10
cargo bench -p atlas-storage --bench hardening -- search/hybrid --save-baseline p2-stable-focused --warm-up-time 3 --measurement-time 5 --sample-size 10
```
