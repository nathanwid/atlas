//! Hardening Phase 2 benchmarks: ingestion throughput, search latency
//! (query-only vs rebuild-plus-query vs full workflows), on the in-memory
//! engine with a deterministic fake embedder.
//!
//! What each measurement includes is labeled in its name and documented in
//! `docs/BENCHMARKS.md`. Fixture generation is deterministic (LCG word
//! selection) and happens in setup, excluded from timing.

use atlas_embedding::Embedder;
use atlas_storage::{BruteForceIndex, VectorIndex};
use atlas_storage::{HybridConfig, SearchMode, Store, hybrid_search, learn, search};
use atlas_tokenizer::BpeTokenizer;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const MODEL: &str = "bench-fake";
const DIM: usize = 512;
const DOC_WORDS: usize = 30; // ~30 tokens: one chunk per document

/// Deterministic fake embedder: bag-of-words into DIM buckets, normalized
/// away (raw counts); identical text yields an identical vector.
struct FakeEmbedder;

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        MODEL
    }

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0; DIM];
                for word in text.split_whitespace() {
                    v[hash_word(word)] += 1.0;
                }
                v
            })
            .collect())
    }
}

fn hash_word(word: &str) -> usize {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in word.to_lowercase().bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash % DIM as u64) as usize
}

fn tokenizer() -> BpeTokenizer {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
    );
    let f = std::fs::File::open(path).expect("ranks fixture");
    BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("ranks parse")
}

/// Deterministic document text; the global counter guarantees uniqueness
/// across iterations so `learn` never short-circuits as a duplicate.
static DOC_COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: [&str; 16] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi",
];

fn unique_doc() -> String {
    let mut state = DOC_COUNTER
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_mul(0x9E3779B97F4A7C15)
        | 1;
    let mut words = Vec::with_capacity(DOC_WORDS);
    for _ in 0..DOC_WORDS {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        words.push(VOCAB[(state % VOCAB.len() as u64) as usize]);
    }
    words.join(" ")
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().expect("tokio runtime")
}

/// Ingestion throughput: fresh in-memory store (setup, untimed) + learning
/// N unique documents (timed, includes chunking, fake embedding, graph
/// extraction, and all persistence upserts).
fn bench_ingestion(c: &mut Criterion) {
    let rt = runtime();
    let tok = tokenizer();
    let embedder = FakeEmbedder;

    let mut group = c.benchmark_group("ingestion");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    for n in [10usize, 100, 1000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("{n}_docs"), |b| {
            b.iter_batched(
                || {
                    let store = rt.block_on(async { Store::open_mem().await.unwrap() });
                    let docs: Vec<String> = (0..n).map(|_| unique_doc()).collect();
                    (store, docs)
                },
                |(store, docs)| {
                    rt.block_on(async {
                        for doc in &docs {
                            learn(&store, &tok, &embedder, doc).await.unwrap();
                        }
                    })
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

/// Search benchmarks over N one-chunk documents.
///
/// * `query_only` — in-memory index search alone (no load, no rebuild).
/// * `load_rebuild_query` — embedding-table load + index rebuild + search.
/// * `workflow_search` — the full M07 path (load + rebuild + embed_query +
///   chunk hydration).
/// * `hybrid_search` — the full M08 path (everything above + graph load +
///   personalized PageRank + RRF fusion).
fn bench_search(c: &mut Criterion) {
    let rt = runtime();
    let tok = tokenizer();
    let embedder = FakeEmbedder;

    let mut group = c.benchmark_group("search");
    group.sample_size(20);
    for n in [100usize, 1000] {
        // Setup (untimed): learn N one-chunk documents, snapshot vectors.
        let store = rt.block_on(async { Store::open_mem().await.unwrap() });
        for _ in 0..n {
            let doc = unique_doc();
            rt.block_on(async { learn(&store, &tok, &embedder, &doc).await.unwrap() });
        }
        let embeddings = rt.block_on(async { store.load_embeddings(MODEL).await.unwrap() });
        assert_eq!(embeddings.len(), n, "one chunk per document");
        let dim = embeddings[0].vector.len();
        let vectors: Vec<Vec<f32>> = embeddings.iter().map(|e| e.vector.clone()).collect();
        let index = BruteForceIndex::build(dim, &vectors).expect("index builds");
        let query_vector = embedder
            .embed_query("alpha beta gamma")
            .expect("query embeds");

        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("query_only/{n}_chunks"), |b| {
            b.iter(|| {
                // In-memory exact scan only: no load, no rebuild, no I/O.
                VectorIndex::search(&index, &query_vector, 5)
            })
        });
        group.bench_function(format!("load_rebuild_query/{n}_chunks"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    // Includes: embedding-table full scan + index rebuild.
                    let embeddings = store.load_embeddings(MODEL).await.unwrap();
                    let dim = embeddings[0].vector.len();
                    let vectors: Vec<Vec<f32>> =
                        embeddings.iter().map(|e| e.vector.clone()).collect();
                    let index = BruteForceIndex::build(dim, &vectors).unwrap();
                    VectorIndex::search(&index, &query_vector, 5)
                })
            })
        });
        group.bench_function(format!("workflow_search/{n}_chunks"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    // Full M07 path: load + rebuild + embed_query + hydration.
                    search(&store, &embedder, "alpha beta gamma", 5)
                        .await
                        .unwrap()
                })
            })
        });
        group.bench_function(format!("hybrid_search_full/{n}_chunks"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    // Full M08 path: workflow_search + graph load + PPR + RRF.
                    hybrid_search(
                        &store,
                        &embedder,
                        "alpha beta gamma",
                        5,
                        &HybridConfig::default(),
                        SearchMode::Hybrid,
                    )
                    .await
                    .unwrap()
                })
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ingestion, bench_search);
criterion_main!(benches);
