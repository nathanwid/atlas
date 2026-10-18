//! Storage PoC benchmarks: the same ingestion and workflow-search
//! measurements as `hardening.rs`, run against three backends:
//!
//! * `surreal_mem` â€” SurrealDB in-memory engine (the M07/M08 baseline;
//!   durability I/O excluded by construction).
//! * `surrealkv_durable` â€” SurrealDB on embedded SurrealKV, fresh
//!   temporary directory per iteration (durable writes, first time this
//!   path is measured).
//! * `sqlite_durable` â€” SQLite (rusqlite bundled) with WAL +
//!   synchronous=FULL, fresh temporary file per iteration.
//!
//! Fixture generation and store creation happen in setup and are excluded
//! from timing; each timed iteration operates on a fresh store so
//! duplicate short-circuits never fire. The fake embedder isolates
//! pipeline + durability cost from model inference (same methodology as
//! `hardening.rs` / `docs/BENCHMARKS.md`).

use atlas_embedding::Embedder;
use atlas_storage::backend::Backend;
use atlas_storage::{HybridConfig, SearchMode, SqliteStore, Store, hybrid_search, learn, search};
use atlas_tokenizer::BpeTokenizer;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const MODEL: &str = "bench-fake";
const DIM: usize = 512;
const DOC_WORDS: usize = 30; // ~30 tokens: one chunk per document

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

/// One bench store of each engine kind, dispatching through `Backend` so
/// the timed bodies are backend-agnostic.
enum BenchStore {
    SurrealMem(Store),
    SurrealKvDurable(Store),
    SqliteDurable(Box<SqliteStore>),
    /// In-memory SQLite, for isolating SQLite engine cost from its disk I/O.
    SqliteMem(Box<SqliteStore>),
}

impl Backend for BenchStore {
    async fn upsert_document(&self, doc_id: &str, text: &str) -> anyhow::Result<()> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.upsert_document(doc_id, text).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.upsert_document(doc_id, text).await
            }
        }
    }

    async fn document_exists(&self, doc_id: &str) -> anyhow::Result<bool> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.document_exists(doc_id).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.document_exists(doc_id).await
            }
        }
    }

    async fn upsert_chunk(&self, chunk: &atlas_storage::ChunkRecord) -> anyhow::Result<()> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.upsert_chunk(chunk).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => s.upsert_chunk(chunk).await,
        }
    }

    async fn get_chunk(
        &self,
        chunk_id: &str,
    ) -> anyhow::Result<Option<atlas_storage::ChunkRecord>> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.get_chunk(chunk_id).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => s.get_chunk(chunk_id).await,
        }
    }

    async fn delete_chunk(&self, chunk_id: &str) -> anyhow::Result<()> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.delete_chunk(chunk_id).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.delete_chunk(chunk_id).await
            }
        }
    }

    async fn upsert_embedding(
        &self,
        embedding: &atlas_storage::EmbeddingRecord,
    ) -> anyhow::Result<()> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.upsert_embedding(embedding).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.upsert_embedding(embedding).await
            }
        }
    }

    async fn load_embeddings(
        &self,
        model_id: &str,
    ) -> anyhow::Result<Vec<atlas_storage::EmbeddingRecord>> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.load_embeddings(model_id).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.load_embeddings(model_id).await
            }
        }
    }

    async fn upsert_entity(&self, entity: &atlas_storage::EntityRecord) -> anyhow::Result<()> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.upsert_entity(entity).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.upsert_entity(entity).await
            }
        }
    }

    async fn upsert_predicate(
        &self,
        predicate: &atlas_storage::PredicateRecord,
    ) -> anyhow::Result<()> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.upsert_predicate(predicate).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.upsert_predicate(predicate).await
            }
        }
    }

    async fn upsert_relationship(
        &self,
        rel: &atlas_storage::RelationshipRecord,
    ) -> anyhow::Result<()> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.upsert_relationship(rel).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.upsert_relationship(rel).await
            }
        }
    }

    async fn load_graph(&self) -> anyhow::Result<atlas_graph::KnowledgeGraph> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => s.load_graph().await,
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => s.load_graph().await,
        }
    }

    async fn apply_learn_write(&self, write: &atlas_storage::LearnWrite<'_>) -> anyhow::Result<()> {
        match self {
            BenchStore::SurrealMem(s) | BenchStore::SurrealKvDurable(s) => {
                s.apply_learn_write(write).await
            }
            BenchStore::SqliteDurable(s) | BenchStore::SqliteMem(s) => {
                s.apply_learn_write(write).await
            }
        }
    }
}

/// Open a fresh store of `engine` in its own temporary directory (kept
/// alive by the guard) â€” setup cost, never timed.
fn open_fresh(
    rt: &tokio::runtime::Runtime,
    engine: &str,
) -> (BenchStore, Option<tempfile::TempDir>) {
    match engine {
        "surreal_mem" => (
            BenchStore::SurrealMem(rt.block_on(Store::open_mem()).unwrap()),
            None,
        ),
        "surrealkv_durable" => {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let store = rt
                .block_on(Store::open_durable(dir.path().join("surreal")))
                .unwrap();
            (BenchStore::SurrealKvDurable(store), Some(dir))
        }
        "sqlite_durable" => {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let store = rt
                .block_on(SqliteStore::open_durable(dir.path().join("atlas.db")))
                .unwrap();
            (BenchStore::SqliteDurable(Box::new(store)), Some(dir))
        }
        "sqlite_mem" => (
            BenchStore::SqliteMem(Box::new(rt.block_on(SqliteStore::open_mem()).unwrap())),
            None,
        ),
        other => panic!("unknown engine {other}"),
    }
}

const ENGINES: [&str; 4] = [
    "surreal_mem",
    "surrealkv_durable",
    "sqlite_mem",
    "sqlite_durable",
];

/// Ingestion throughput per engine: fresh store (setup, untimed) + learning
/// 100 unique documents (timed, includes chunking, fake embedding, graph
/// extraction, and all persistence upserts â€” including durability I/O for
/// the durable engines).
fn bench_durable_ingestion(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let tok = tokenizer();
    let embedder = FakeEmbedder;
    let n = 100usize;

    let mut group = c.benchmark_group("durable_ingestion");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    for engine in ENGINES {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("{engine}/{n}_docs"), |b| {
            b.iter_batched(
                || {
                    let (store, _guard) = open_fresh(&rt, engine);
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

/// Search per engine: the full M07 workflow path (embedding-table load +
/// index rebuild + embed_query + hydration) and the full M08 hybrid path
/// (plus graph load + PPR + RRF), over 100 one-chunk documents in a
/// durable store written once in setup.
fn bench_durable_search(c: &mut Criterion) {
    let tok = tokenizer();
    let embedder = FakeEmbedder;
    let n = 100usize;

    let mut group = c.benchmark_group("durable_search");
    group.sample_size(10);
    for engine in ENGINES {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let (store, _guard) = open_fresh(&rt, engine);
        rt.block_on(async {
            for _ in 0..n {
                let doc = unique_doc();
                learn(&store, &tok, &embedder, &doc).await.unwrap();
            }
        });

        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("{engine}/workflow_search/{n}_chunks"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    search(&store, &embedder, "alpha beta gamma", 5)
                        .await
                        .unwrap()
                })
            })
        });
        group.bench_function(format!("{engine}/hybrid_search_full/{n}_chunks"), |b| {
            b.iter(|| {
                rt.block_on(async {
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

criterion_group!(benches, bench_durable_ingestion, bench_durable_search);
criterion_main!(benches);
