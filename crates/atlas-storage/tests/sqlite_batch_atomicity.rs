#![cfg(feature = "sqlite-backend")]
//! Transaction-batching atomicity for the SQLite backend (storage PoC).
//!
//! `learn` persists one document through a single grouped transaction
//! (`apply_learn_write`). These tests pin the two properties that matter:
//!
//! 1. **Rollback**: if any write inside the group fails, the whole group
//!    is rolled back â€” no document, chunk, embedding, entity, predicate,
//!    relationship, or evidence row from the failed `learn` may survive.
//! 2. **Success**: a valid group lands completely, and re-learning is a
//!    clean duplicate.
//!
//! The failure is injected deterministically at the SQL level: the test
//! rebuilds the `predicate` table with a `CHECK` constraint that rejects
//! one specific key, so the batch fails exactly at the predicate write,
//! after the document/chunk/embedding writes have already run inside the
//! open transaction.

use atlas_embedding::Embedder;
use atlas_storage::backend::Backend;
use atlas_storage::{DocumentRecord, LearnWrite, SqliteStore, doc_id_for, learn};
use atlas_tokenizer::BpeTokenizer;
use std::time::Duration;

struct FakeEmbedder;

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        "batch-atomicity-model"
    }

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
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

/// Rebuild `predicate` so that inserting the key `poison` fails with a
/// CHECK violation while every other key keeps working. Runs on a raw
/// connection after the store handle is dropped; the store's
/// `CREATE TABLE IF NOT EXISTS` is a no-op on the tampered table.
fn poison_predicate_table(db: &std::path::Path) {
    let mut conn = rusqlite::Connection::open(db).expect("reopen raw");
    conn.busy_timeout(Duration::from_millis(5000)).unwrap();
    let tx = conn.transaction().unwrap();
    tx.execute_batch(
        "CREATE TABLE predicate_poisoned (
             id TEXT PRIMARY KEY,
             value TEXT NOT NULL,
             CHECK (id != 'poison')
         );
         INSERT INTO predicate_poisoned(id, value) SELECT id, value FROM predicate;
         DROP TABLE predicate;
         ALTER TABLE predicate_poisoned RENAME TO predicate;",
    )
    .unwrap();
    tx.commit().unwrap();
}

/// A small graph whose predicate interning produces the key `poison`.
fn poisoned_graph() -> atlas_graph::KnowledgeGraph {
    let mut graph = atlas_graph::KnowledgeGraph::new();
    let src = graph.intern_entity("Poison");
    let pid = graph.intern_predicate("poison");
    let dst = graph.intern_entity("Poison");
    graph.link(
        src,
        pid,
        dst,
        atlas_graph::EdgeOrigin::Explicit,
        atlas_graph::Evidence {
            doc_id: "batch-atomicity-test".to_string(),
            sentence: "Poison poison Poison.".to_string(),
        },
    );
    graph
}

#[tokio::test]
async fn failed_group_rolls_back_every_write_it_contained() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("atlas.db");

    // Prime + tamper.
    {
        let store = SqliteStore::open_durable(&db).await.unwrap();
        let summary = learn(&store, &tokenizer(), &FakeEmbedder, "seed document text")
            .await
            .unwrap();
        assert!(!summary.duplicate);
    }
    poison_predicate_table(&db);

    let store = SqliteStore::open_durable(&db).await.unwrap();

    // One batch whose predicate write fails mid-transaction.
    let text = "batch atomicity probe document";
    let doc_id = doc_id_for(text);
    let document = DocumentRecord {
        doc_id: doc_id.clone(),
        text: text.to_string(),
    };
    let chunk = atlas_storage::ChunkRecord {
        chunk_id: format!("{doc_id}:0"),
        doc_id: doc_id.clone(),
        ordinal: 0,
        text: text.to_string(),
    };
    let embedding = atlas_storage::EmbeddingRecord {
        chunk_id: chunk.chunk_id.clone(),
        model_id: "batch-atomicity-model".to_string(),
        vector: vec![1.0, 2.0, 3.0],
    };
    let write = LearnWrite {
        document: &document,
        chunks: std::slice::from_ref(&chunk),
        embeddings: std::slice::from_ref(&embedding),
        graph: &poisoned_graph(),
    };

    let result = store.apply_learn_write(&write).await;
    assert!(
        result.is_err(),
        "the poisoned predicate must fail the batch"
    );

    // Nothing from the group survived: the document write ran first inside
    // the transaction, but the later failure rolls it back too.
    assert!(
        !store.document_exists(&doc_id).await.unwrap(),
        "document row from the failed group must be rolled back"
    );
    assert!(
        store.get_chunk(&chunk.chunk_id).await.unwrap().is_none(),
        "chunk row from the failed group must be rolled back"
    );
    let embeddings = store
        .load_embeddings("batch-atomicity-model")
        .await
        .unwrap();
    assert!(
        embeddings.iter().all(|e| e.chunk_id != chunk.chunk_id),
        "embedding row from the failed group must be rolled back"
    );
    let graph = store.load_graph().await.unwrap();
    assert!(
        graph.relationship_count() == 0,
        "relationship/evidence from the failed group must be rolled back"
    );
    assert!(
        graph.find_entity("poison").is_none(),
        "entity from the failed group must be rolled back"
    );

    // The store is still healthy: a normal learn succeeds and is a clean
    // duplicate afterwards.
    let summary = learn(
        &store,
        &tokenizer(),
        &FakeEmbedder,
        "healthy learn afterwards",
    )
    .await
    .unwrap();
    assert!(!summary.duplicate);
    let again = learn(
        &store,
        &tokenizer(),
        &FakeEmbedder,
        "healthy learn afterwards",
    )
    .await
    .unwrap();
    assert!(again.duplicate);
}

#[tokio::test]
async fn successful_group_lands_completely_and_is_idempotent() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("atlas.db");
    let store = SqliteStore::open_durable(&db).await.unwrap();

    let text = "Rust uses Tokio for async runtimes.";
    let summary = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    assert!(!summary.duplicate);
    let doc_id = doc_id_for(text);

    // Everything the group promised is present.
    assert!(store.document_exists(&doc_id).await.unwrap());
    assert!(
        store
            .get_chunk(&format!("{doc_id}:0"))
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        store
            .load_embeddings("batch-atomicity-model")
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        store.load_graph().await.unwrap().relationship_count() >= 1,
        "fixture-style sentence must extract a relationship"
    );

    // Idempotent re-learning: clean duplicate, no partial state.
    let again = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    assert!(again.duplicate);
    assert_eq!(
        store
            .load_embeddings("batch-atomicity-model")
            .await
            .unwrap()
            .len(),
        1
    );
}
