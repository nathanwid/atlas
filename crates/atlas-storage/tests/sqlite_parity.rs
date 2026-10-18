#![cfg(feature = "sqlite-backend")]
//! Backend parity tests (storage PoC).
//!
//! The same pipeline (`learn` / `search` / `hybrid_search`) is run against
//! the SurrealDB `Store` and the SQLite `SqliteStore`, over the existing
//! eight-document fixture. Everything asserted here must match across
//! backends: ingestion summaries, duplicate detection, partial-write
//! repair, deleted-chunk degradation, restart persistence (durable
//! backends), and retrieval results.
//!
//! Scores are compared with a 1e-6 tolerance: both backends feed identical
//! f32 vectors into identical index code, so scores should be bitwise
//! equal, but the comparison is loosened to absorb any cross-platform
//! float-lib noise. Ranking relies on distinct vectors (no score ties), so
//! row-order differences between engines cannot change the top-k order.

use atlas_embedding::Embedder;
use atlas_storage::backend::Backend;
use atlas_storage::{
    ChunkRecord, HybridConfig, HybridResult, LearnSummary, SearchMode, SearchResult, SqliteStore,
    Store, doc_id_for, hybrid_search, learn, search,
};
use atlas_tokenizer::BpeTokenizer;
use std::collections::HashSet;

const FIXTURE: &str = include_str!("fixtures/hybrid_docs.txt");

/// Deterministic word-overlap embedder (same shape as the M08 tests).
struct WordOverlapEmbedder;

const DIM: usize = 512;

fn hash_word(word: &str) -> usize {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in word.to_lowercase().bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash % DIM as u64) as usize
}

impl Embedder for WordOverlapEmbedder {
    fn model_id(&self) -> &str {
        "word-overlap"
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

fn tokenizer() -> BpeTokenizer {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
    );
    let f = std::fs::File::open(path).expect("ranks fixture");
    BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("ranks parse")
}

/// Stable, order-independent graph signature (same construction as
/// `restart_recovery.rs`): sorted entity rows and sorted relationship rows
/// keyed by normalized identity.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
struct EvidenceRow(String, String);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
struct RelationshipRow(String, String, String, String, Vec<EvidenceRow>);

#[derive(Debug, PartialEq)]
struct GraphSignature {
    entities: Vec<(String, String, Option<String>)>,
    relationships: Vec<RelationshipRow>,
}

fn signature(graph: &atlas_graph::KnowledgeGraph) -> GraphSignature {
    let mut entities: Vec<_> = graph
        .entities()
        .map(|(id, e)| {
            (
                graph.entity_key(id).unwrap_or_default(),
                e.display.clone(),
                e.kind.clone(),
            )
        })
        .collect();
    entities.sort();

    let mut relationships: Vec<RelationshipRow> = graph
        .relationships()
        .map(|(_, r)| {
            RelationshipRow(
                graph.entity_key(r.src).unwrap_or_default(),
                graph.predicate(r.predicate).unwrap_or_default().to_string(),
                graph.entity_key(r.dst).unwrap_or_default(),
                format!("{:?}", r.origin),
                {
                    let mut ev: Vec<EvidenceRow> = r
                        .evidence
                        .iter()
                        .map(|e| EvidenceRow(e.doc_id.clone(), e.sentence.clone()))
                        .collect();
                    ev.sort();
                    ev
                },
            )
        })
        .collect();
    relationships.sort();

    GraphSignature {
        entities,
        relationships,
    }
}

/// Compare the score-carrying prefix (score > 0) of two result vectors:
/// identical membership, order, scores, and chunk text.
///
/// Zero-overlap chunks all score exactly 0.0; which of them fill the
/// remaining top-k slots is decided by index row order, i.e. each engine's
/// unspecified row order (SurrealDB select order vs SQLite rowid). Tie-band
/// membership is therefore engine-dependent by design and excluded from
/// parity; the overlap queries used here make the scored prefix strictly
/// ordered and engine-independent.
fn assert_scored_prefix_match(a: &[SearchResult], b: &[SearchResult], what: &str) {
    let pa: Vec<&SearchResult> = a.iter().filter(|r| r.score > 0.0).collect();
    let pb: Vec<&SearchResult> = b.iter().filter(|r| r.score > 0.0).collect();
    assert_eq!(pa.len(), pb.len(), "{what}: scored-result count differs");
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!(x.chunk_id, y.chunk_id, "{what}: ranking differs");
        assert_eq!(x.doc_id, y.doc_id, "{what}: doc ids differ");
        assert_eq!(x.text, y.text, "{what}: chunk text differs");
        assert!(
            (x.score - y.score).abs() < 1e-6,
            "{what}: score {} vs {}",
            x.score,
            y.score
        );
    }
    assert!(!pa.is_empty(), "{what}: query must match something");
    assert_eq!(a[0].chunk_id, b[0].chunk_id, "{what}: top-1 differs");
}

/// Hybrid parity on the vector-score-carrying tier (score is `Some(>0)`):
/// identical membership, order, scores, and provenance.
///
/// Results sourced only from exact-score ties or the graph expansion over
/// tie-band seeds inherit engine row order and are excluded, for the same
/// reason as [`assert_scored_prefix_match`]. Same-engine determinism and
/// graph identity are asserted separately and strictly.
fn assert_hybrid_scored_prefix_match(a: &[HybridResult], b: &[HybridResult], what: &str) {
    let pa: Vec<&HybridResult> = a
        .iter()
        .filter(|r| matches!(r.score, Some(s) if s > 0.0))
        .collect();
    let pb: Vec<&HybridResult> = b
        .iter()
        .filter(|r| matches!(r.score, Some(s) if s > 0.0))
        .collect();
    assert_eq!(pa.len(), pb.len(), "{what}: scored-result count differs");
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!(x.chunk_id, y.chunk_id, "{what}: ranking differs");
        assert_eq!(x.source, y.source, "{what}: provenance differs");
        assert_eq!(x.via, y.via, "{what}: provenance path differs");
        let (p, q) = (x.score.unwrap(), y.score.unwrap());
        assert!((p - q).abs() < 1e-6, "{what}: score {p} vs {q}");
    }
    assert!(!pa.is_empty(), "{what}: query must match something");
    assert_eq!(a[0].chunk_id, b[0].chunk_id, "{what}: top-1 differs");
}

#[tokio::test]
async fn learn_summaries_and_duplicate_detection_match_across_backends() {
    let tok = tokenizer();
    let embedder = WordOverlapEmbedder;

    async fn run<S: Backend>(
        store: &S,
        tok: &BpeTokenizer,
        embedder: &WordOverlapEmbedder,
    ) -> (Vec<LearnSummary>, Vec<LearnSummary>) {
        let mut first = Vec::new();
        for doc in FIXTURE.lines() {
            first.push(learn(store, tok, embedder, doc).await.unwrap());
        }
        let mut second = Vec::new();
        for doc in FIXTURE.lines() {
            second.push(learn(store, tok, embedder, doc).await.unwrap());
        }
        (first, second)
    }

    let surreal = Store::open_mem().await.unwrap();
    let (s_first, s_second) = run(&surreal, &tok, &embedder).await;
    let sqlite = SqliteStore::open_mem().await.unwrap();
    let (q_first, q_second) = run(&sqlite, &tok, &embedder).await;

    assert_eq!(s_first.len(), FIXTURE.lines().count());
    assert_eq!(
        s_first, q_first,
        "ingestion summaries must be identical across backends"
    );
    assert!(
        s_first.iter().all(|s| !s.duplicate && s.chunks == 1),
        "each fixture document is one chunk"
    );
    assert_eq!(
        s_second, q_second,
        "duplicate-detection summaries must be identical across backends"
    );
    assert!(
        s_second.iter().all(|s| s.duplicate),
        "re-learning the fixture must short-circuit as duplicates"
    );
}

#[tokio::test]
async fn partial_write_repair_matches_across_backends() {
    async fn run<S: Backend>(
        store: &S,
        tok: &BpeTokenizer,
        embedder: &WordOverlapEmbedder,
    ) -> (LearnSummary, LearnSummary) {
        let text = "alpha beta gamma delta epsilon zeta eta theta";
        let doc_id = doc_id_for(text);
        // Simulate a partial failure: document row written, no chunks.
        store.upsert_document(&doc_id, text).await.unwrap();
        assert!(
            store
                .get_chunk(&format!("{doc_id}:0"))
                .await
                .unwrap()
                .is_none()
        );
        let repaired = learn(store, tok, embedder, text).await.unwrap();
        assert!(!repaired.duplicate);
        let duplicate = learn(store, tok, embedder, text).await.unwrap();
        (repaired, duplicate)
    }

    let surreal = Store::open_mem().await.unwrap();
    let (s_repaired, s_dup) = run(&surreal, &tokenizer(), &WordOverlapEmbedder).await;
    let sqlite = SqliteStore::open_mem().await.unwrap();
    let (q_repaired, q_dup) = run(&sqlite, &tokenizer(), &WordOverlapEmbedder).await;

    assert_eq!(s_repaired, q_repaired);
    assert_eq!(s_repaired.chunks, 1);
    assert_eq!(s_dup, q_dup);
    assert!(s_dup.duplicate);
}

#[tokio::test]
async fn missing_later_chunks_repair_matches_across_backends() {
    async fn run<S: Backend>(
        store: &S,
        tok: &BpeTokenizer,
        embedder: &WordOverlapEmbedder,
    ) -> usize {
        let text = "filler ".repeat(600);
        let doc_id = doc_id_for(&text);
        let first = learn(store, tok, embedder, &text).await.unwrap();
        assert!(first.chunks > 2);
        store
            .delete_chunk(&format!("{doc_id}:{}", first.chunks - 1))
            .await
            .unwrap();
        let repaired = learn(store, tok, embedder, &text).await.unwrap();
        assert!(!repaired.duplicate, "missing chunk must trigger repair");
        for ordinal in 0..repaired.chunks {
            assert!(
                store
                    .get_chunk(&format!("{doc_id}:{ordinal}"))
                    .await
                    .unwrap()
                    .is_some(),
                "chunk {ordinal} must exist after repair"
            );
        }
        repaired.chunks
    }

    let surreal = Store::open_mem().await.unwrap();
    let s_chunks = run(&surreal, &tokenizer(), &WordOverlapEmbedder).await;
    let sqlite = SqliteStore::open_mem().await.unwrap();
    let q_chunks = run(&sqlite, &tokenizer(), &WordOverlapEmbedder).await;
    assert_eq!(s_chunks, q_chunks);
}

#[tokio::test]
async fn deleted_chunk_degrades_search_identically_across_backends() {
    async fn run<S: Backend>(
        store: &S,
        tok: &BpeTokenizer,
        embedder: &WordOverlapEmbedder,
    ) -> (Vec<SearchResult>, Vec<HybridResult>) {
        for doc in FIXTURE.lines() {
            learn(store, tok, embedder, doc).await.unwrap();
        }
        // Delete one chunk out from under its embedding (torn-write shape).
        let victim = format!("{}:0", doc_id_for("SurrealDB stores documents durably."));
        store.delete_chunk(&victim).await.unwrap();

        // Query with real word overlap so the top-1 has a distinct score;
        // zero-overlap queries tie at 0.0 across every chunk, and tie order
        // follows engine row order (not asserted as parity).
        let vector = search(store, embedder, "tokio async runtimes", 5)
            .await
            .unwrap();
        assert!(
            vector.iter().all(|r| r.chunk_id != victim),
            "deleted chunk must not appear in vector results"
        );
        let hybrid = hybrid_search(
            store,
            embedder,
            "tokio async runtimes",
            5,
            &HybridConfig::default(),
            SearchMode::Hybrid,
        )
        .await
        .unwrap();
        assert!(
            hybrid.iter().all(|r| r.chunk_id != victim),
            "deleted chunk must not appear in hybrid results"
        );
        (vector, hybrid)
    }

    let surreal = Store::open_mem().await.unwrap();
    let (s_vec, s_hybrid) = run(&surreal, &tokenizer(), &WordOverlapEmbedder).await;
    let sqlite = SqliteStore::open_mem().await.unwrap();
    let (q_vec, q_hybrid) = run(&sqlite, &tokenizer(), &WordOverlapEmbedder).await;

    assert_scored_prefix_match(&s_vec, &q_vec, "vector search after delete");
    assert_hybrid_scored_prefix_match(&s_hybrid, &q_hybrid, "hybrid search after delete");
}

#[tokio::test]
async fn whitespace_and_empty_edge_cases_match_across_backends() {
    async fn run<S: Backend>(
        store: &S,
        tok: &BpeTokenizer,
        embedder: &WordOverlapEmbedder,
    ) -> (LearnSummary, Option<String>, bool) {
        let empty = learn(store, tok, embedder, "").await.unwrap();
        let empty_exists = store.document_exists(&doc_id_for("")).await.unwrap();
        let ws = learn(store, tok, embedder, "   ").await.unwrap();
        let ws_chunk: Option<ChunkRecord> = store
            .get_chunk(&format!("{}:0", doc_id_for("   ")))
            .await
            .unwrap();
        (
            empty,
            ws_chunk.map(|c| c.text),
            empty_exists && ws.chunks == 1,
        )
    }

    let surreal = Store::open_mem().await.unwrap();
    let s = run(&surreal, &tokenizer(), &WordOverlapEmbedder).await;
    let sqlite = SqliteStore::open_mem().await.unwrap();
    let q = run(&sqlite, &tokenizer(), &WordOverlapEmbedder).await;
    assert_eq!(s, q);
    assert_eq!(s.0.chunks, 0, "empty text is a chunkless document");
    assert_eq!(s.1.as_deref(), Some("   "), "whitespace is content");
}

#[tokio::test]
async fn restart_persistence_matches_across_durable_backends() {
    let tok = tokenizer();
    let embedder = WordOverlapEmbedder;

    // Durable SurrealKV run.
    let surreal_dir = tempfile::TempDir::new().unwrap();
    let before_surreal;
    {
        let store = Store::open_durable(surreal_dir.path().join("surreal"))
            .await
            .unwrap();
        for doc in FIXTURE.lines() {
            learn(&store, &tok, &embedder, doc).await.unwrap();
        }
        before_surreal = signature(&store.load_graph().await.unwrap());
    }
    let surreal_reopened = Store::open_durable(surreal_dir.path().join("surreal"))
        .await
        .unwrap();
    let after_surreal = signature(&surreal_reopened.load_graph().await.unwrap());
    assert_eq!(
        before_surreal, after_surreal,
        "surreal graph survives restart"
    );
    // k = 2: both matching chunks have distinct scores. Larger k would cut
    // into the exact-zero tie band, feed an engine-ordered seed to the PPR
    // stage, and make even the fused top-1 legitimately engine-dependent.
    let s_hybrid = hybrid_search(
        &surreal_reopened,
        &embedder,
        "python bindings",
        2,
        &HybridConfig::default(),
        SearchMode::Hybrid,
    )
    .await
    .unwrap();
    assert!(!s_hybrid.is_empty(), "retrieval must work after restart");

    // Durable SQLite run.
    let sqlite_dir = tempfile::TempDir::new().unwrap();
    let before_sqlite;
    {
        let store = SqliteStore::open_durable(sqlite_dir.path().join("atlas.db"))
            .await
            .unwrap();
        for doc in FIXTURE.lines() {
            learn(&store, &tok, &embedder, doc).await.unwrap();
        }
        before_sqlite = signature(&store.load_graph().await.unwrap());
    }
    let sqlite_reopened = SqliteStore::open_durable(sqlite_dir.path().join("atlas.db"))
        .await
        .unwrap();
    let after_sqlite = signature(&sqlite_reopened.load_graph().await.unwrap());
    assert_eq!(before_sqlite, after_sqlite, "sqlite graph survives restart");

    // Cross-backend: the persisted graphs are identical.
    assert_eq!(
        before_surreal, before_sqlite,
        "persisted graphs must be identical across durable backends"
    );

    // Cross-backend: hybrid retrieval agrees after restart.
    let q_hybrid = hybrid_search(
        &sqlite_reopened,
        &embedder,
        "python bindings",
        2,
        &HybridConfig::default(),
        SearchMode::Hybrid,
    )
    .await
    .unwrap();
    assert_hybrid_scored_prefix_match(&s_hybrid, &q_hybrid, "hybrid across durable backends");

    // Determinism within a backend: repeated queries agree.
    let ids =
        |r: &[HybridResult]| -> Vec<String> { r.iter().map(|h| h.chunk_id.clone()).collect() };
    let again = hybrid_search(
        &sqlite_reopened,
        &embedder,
        "python bindings",
        2,
        &HybridConfig::default(),
        SearchMode::Hybrid,
    )
    .await
    .unwrap();
    assert_eq!(ids(&q_hybrid), ids(&again));
    let unique: HashSet<String> = ids(&q_hybrid).into_iter().collect();
    assert_eq!(unique.len(), q_hybrid.len(), "no duplicate results");
}
