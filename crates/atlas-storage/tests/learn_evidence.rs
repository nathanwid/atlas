//! Incremental learning must retain evidence from earlier documents.

use atlas_graph::{EdgeOrigin, Evidence, KnowledgeGraph};
use atlas_storage::{
    Backend, ChunkRecord, DocumentRecord, EvidenceRecord, LearnWrite, RelationshipRecord,
    SqliteStore, doc_id_for,
};

async fn apply_fact(store: &SqliteStore, text: &str, origin: EdgeOrigin) -> anyhow::Result<()> {
    let doc_id = doc_id_for(text);
    let document = DocumentRecord {
        doc_id: doc_id.clone(),
        text: text.into(),
    };
    let chunk = ChunkRecord {
        chunk_id: format!("{doc_id}:0"),
        doc_id,
        ordinal: 0,
        text: text.into(),
    };
    let mut graph = KnowledgeGraph::new();
    let rust = graph.intern_entity("Rust");
    let tokio = graph.intern_entity("Tokio");
    let uses = graph.intern_predicate("uses");
    graph.link(
        rust,
        uses,
        tokio,
        origin,
        Evidence {
            doc_id: chunk.chunk_id.clone(),
            sentence: text.into(),
        },
    );
    store
        .apply_learn_write(&LearnWrite {
            document: &document,
            chunks: &[chunk],
            embeddings: &[],
            graph: &graph,
        })
        .await
}

async fn evidence(store: &SqliteStore) -> Vec<String> {
    let graph = store.load_graph().await.unwrap();
    assert_eq!(graph.relationship_count(), 1);
    graph
        .relationships()
        .next()
        .unwrap()
        .1
        .evidence
        .iter()
        .map(|ev| ev.sentence.clone())
        .collect()
}

#[tokio::test]
async fn grouped_writes_merge_evidence_without_duplicates_and_survive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("atlas.db");
    let first = "Rust uses Tokio.";
    let second = "Rust uses Tokio for async runtimes.";
    {
        let store = SqliteStore::open_durable(&path).await.unwrap();
        apply_fact(&store, first, EdgeOrigin::Explicit)
            .await
            .unwrap();
        apply_fact(&store, second, EdgeOrigin::Explicit)
            .await
            .unwrap();
        apply_fact(&store, second, EdgeOrigin::Explicit)
            .await
            .unwrap();
        assert_eq!(evidence(&store).await, [first, second]);
    }
    let store = SqliteStore::open_durable(&path).await.unwrap();
    assert_eq!(evidence(&store).await, [first, second]);
}

#[tokio::test]
async fn merge_preserves_origin_and_ordinal_but_direct_upsert_still_replaces() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("atlas.db");
    let store = SqliteStore::open_durable(&path).await.unwrap();
    apply_fact(&store, "first", EdgeOrigin::Explicit)
        .await
        .unwrap();
    let replacement = RelationshipRecord {
        src: "rust".into(),
        predicate: "uses".into(),
        dst: "tokio".into(),
        origin: "explicit".into(),
        ordinal: 17,
        evidence: vec![EvidenceRecord {
            doc_id: "old:0".into(),
            sentence: "original".into(),
        }],
    };
    store.upsert_relationship(&replacement).await.unwrap();
    apply_fact(&store, "second", EdgeOrigin::CoOccurrence)
        .await
        .unwrap();
    assert_eq!(evidence(&store).await, ["original", "second"]);
    let raw = rusqlite::Connection::open(&path).unwrap();
    let metadata: (String, u32) = raw
        .query_row("SELECT origin, ordinal FROM relationship", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(metadata, ("explicit".into(), 17));
    store.upsert_relationship(&replacement).await.unwrap();
    assert_eq!(evidence(&store).await, ["original"]);
}

#[tokio::test]
async fn failed_evidence_merge_restores_previous_evidence_and_rolls_back_document() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("atlas.db");
    let store = SqliteStore::open_durable(&path).await.unwrap();
    apply_fact(&store, "original", EdgeOrigin::Explicit)
        .await
        .unwrap();
    let raw = rusqlite::Connection::open(&path).unwrap();
    raw.execute_batch(
        "CREATE TRIGGER reject_evidence BEFORE INSERT ON evidence
        WHEN NEW.sentence = 'rejected'
        BEGIN SELECT RAISE(ABORT, 'evidence failure'); END;",
    )
    .unwrap();
    assert!(
        apply_fact(&store, "rejected", EdgeOrigin::Explicit)
            .await
            .is_err()
    );
    assert_eq!(evidence(&store).await, ["original"]);
    let id = doc_id_for("rejected");
    assert!(!store.document_exists(&id).await.unwrap());
    assert!(store.get_chunk(&format!("{id}:0")).await.unwrap().is_none());
}

#[test]
fn two_connections_keep_both_committed_sources() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("atlas.db");
    let rt = tokio::runtime::Runtime::new().unwrap();
    let stores = [
        rt.block_on(SqliteStore::open_durable(&path)).unwrap(),
        rt.block_on(SqliteStore::open_durable(&path)).unwrap(),
    ];
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let handles: Vec<_> = stores
        .into_iter()
        .zip(["first", "second"])
        .map(|(store, text)| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                barrier.wait();
                rt.block_on(apply_fact(&store, text, EdgeOrigin::Explicit))
                    .unwrap();
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let store = rt.block_on(SqliteStore::open_durable(&path)).unwrap();
    let mut sources = rt.block_on(evidence(&store));
    sources.sort();
    assert_eq!(sources, ["first", "second"]);
}
