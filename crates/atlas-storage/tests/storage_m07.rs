//! M07 storage tests: persistence, idempotency, restart, and the durable
//! learn/search workflows. All offline: `mem` engine + a deterministic fake
//! embedder. The restart test uses the durable SurrealKV engine in a
//! tempdir.

use atlas_embedding::Embedder;
use atlas_storage::{ChunkRecord, Store, doc_id_for, learn, search};
use atlas_tokenizer::BpeTokenizer;
use tempfile::TempDir;

/// Deterministic fake embedder: vector = [text length, starts with 'a'].
/// Identical text embeds to an identical vector, so exact-search assertions
/// are exact.
struct FakeEmbedder;

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        "fake-test-model"
    }

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| vec![t.len() as f32, t.starts_with('a') as u8 as f32])
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

#[tokio::test]
async fn open_mem_starts_empty() {
    let store = Store::open_mem().await.unwrap();
    let doc = doc_id_for("anything");
    assert!(!store.document_exists(&doc).await.unwrap());
    assert!(
        store
            .load_embeddings("fake-test-model")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn learn_persists_document_chunks_embeddings() {
    let store = Store::open_mem().await.unwrap();
    let text = "alpha beta gamma delta epsilon"; // single chunk
    let summary = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();

    assert!(!summary.duplicate);
    assert_eq!(summary.chunks, 1);
    assert_eq!(summary.embeddings, 1);

    let doc_id = doc_id_for(text);
    assert!(store.document_exists(&doc_id).await.unwrap());
    let chunk = store
        .get_chunk(&format!("{doc_id}:0"))
        .await
        .unwrap()
        .expect("chunk 0 exists");
    assert_eq!(chunk.text, text);
    assert_eq!(chunk.ordinal, 0);

    let embeddings = store.load_embeddings("fake-test-model").await.unwrap();
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].chunk_id, format!("{doc_id}:0"));
    assert_eq!(
        embeddings[0].vector,
        vec![text.len() as f32, 1.0],
        "text starts with 'a'"
    );
}

#[tokio::test]
async fn duplicate_learn_is_idempotent() {
    let store = Store::open_mem().await.unwrap();
    let text = "alpha beta gamma delta epsilon";
    let first = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    let second = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();

    assert!(!first.duplicate);
    assert!(second.duplicate);
    assert_eq!(second.chunks, 0);
    assert_eq!(second.embeddings, 0);
    assert_eq!(
        store
            .load_embeddings("fake-test-model")
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn persisted_search_returns_the_learned_chunk() {
    let store = Store::open_mem().await.unwrap();
    let text = "alpha beta gamma delta epsilon";
    learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();

    // Same text as the query: identical fake vector, exact score 1.0.
    let hits = search(&store, &FakeEmbedder, text, 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert!((hits[0].score - 1.0).abs() < 1e-5);
    assert_eq!(hits[0].text, text);

    // Different text: some score below 1, still deterministic.
    let hits = search(&store, &FakeEmbedder, "bbbbbbbbbbbb", 5)
        .await
        .unwrap();
    assert!(hits[0].score < 1.0);
}

#[tokio::test]
async fn graph_roundtrip_preserves_semantics_and_evidence() {
    let store = Store::open_mem().await.unwrap();
    let tok = tokenizer();
    // Two docs: first creates a co-occurrence edge, second upgrades it via
    // the explicit rule and adds its own predicate edge.
    learn(&store, &tok, &FakeEmbedder, "Rust and Cargo shipped.")
        .await
        .unwrap();
    learn(&store, &tok, &FakeEmbedder, "Rust extends Cargo broadly.")
        .await
        .unwrap();

    let graph = store.load_graph().await.unwrap();
    let rust = graph.find_entity("rust").expect("rust entity");
    let cargo = graph.find_entity("cargo").expect("cargo entity");

    // First-seen display labels survive.
    for (id, entity) in graph.entities() {
        let _ = id;
        assert!(entity.kind.is_none(), "extraction never sets kind");
    }
    assert!(graph.entity(rust).unwrap().display.starts_with('R'));

    // The fallback edge was upgraded by the explicit sighting.
    let co: Vec<_> = graph
        .relationships()
        .filter(|(_, r)| graph.predicate(r.predicate) == Some("co_occurs_with"))
        .map(|(_, r)| r)
        .collect();
    assert_eq!(co.len(), 1);
    assert_eq!(co[0].origin, atlas_graph::EdgeOrigin::Explicit);
    assert_eq!(
        co[0].evidence.len(),
        1,
        "fallback skipped for explicitly-linked pairs, so one sighting"
    );
    // Evidence doc_id is the chunk that produced it.
    assert!(
        co[0].evidence[0].doc_id.ends_with(":0"),
        "evidence must reference a chunk, got {:?}",
        co[0].evidence[0].doc_id
    );

    // The explicit edge exists under its own predicate.
    let extends: Vec<_> = graph
        .relationships()
        .filter(|(_, r)| graph.predicate(r.predicate) == Some("extends"))
        .map(|(_, r)| r)
        .collect();
    assert_eq!(extends.len(), 1);
    assert_eq!(extends[0].origin, atlas_graph::EdgeOrigin::Explicit);
    let _ = (rust, cargo);
}

#[tokio::test]
async fn durable_store_survives_restart() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("surreal");
    let text = "alpha beta gamma delta epsilon";

    {
        let store = Store::open_durable(&db_path).await.unwrap();
        learn(&store, &tokenizer(), &FakeEmbedder, text)
            .await
            .unwrap();
        assert!(store.document_exists(&doc_id_for(text)).await.unwrap());
    } // handle dropped: the durable engine must have flushed

    let reopened = Store::open_durable(&db_path).await.unwrap();
    assert!(reopened.document_exists(&doc_id_for(text)).await.unwrap());
    let hits = search(&reopened, &FakeEmbedder, text, 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].text, text);
}

#[tokio::test]
async fn chunk_records_are_recoverable_by_id() {
    let store = Store::open_mem().await.unwrap();
    let text = "alpha beta";
    learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    let doc_id = doc_id_for(text);
    let chunk: ChunkRecord = store
        .get_chunk(&format!("{doc_id}:0"))
        .await
        .unwrap()
        .expect("exists");
    assert_eq!(chunk.doc_id, doc_id);
}
