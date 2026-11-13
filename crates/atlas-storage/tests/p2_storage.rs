use atlas_embedding::Embedder;
use atlas_graph::{EdgeOrigin, Evidence, KnowledgeGraph};
use atlas_storage::{Backend, DocumentRecord, LearnWrite, SqliteStore, doc_id_for, learn, search};
use atlas_tokenizer::BpeTokenizer;
struct Fake(&'static str);
impl Embedder for Fake {
    fn model_id(&self) -> &str {
        self.0
    }
    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                if t.contains("Rust") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.5, 0.5]
                }
            })
            .collect())
    }
}
fn tok() -> BpeTokenizer {
    BpeTokenizer::from_tiktoken_reader(
        std::fs::File::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
        ))
        .unwrap(),
    )
    .unwrap()
}
#[tokio::test]
async fn models_and_removed_derived_records_are_repaired() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("atlas.db");
    let store = SqliteStore::open_durable(&path).await.unwrap();
    let tok = tok();
    let text = "Rust uses Tokio.";
    learn(&store, &tok, &Fake("a"), text).await.unwrap();
    assert!(
        !learn(&store, &tok, &Fake("b"), text)
            .await
            .unwrap()
            .duplicate
    );
    assert_eq!(store.load_embeddings("a").await.unwrap().len(), 1);
    assert_eq!(store.load_embeddings("b").await.unwrap().len(), 1);
    assert!(
        learn(&store, &tok, &Fake("b"), text)
            .await
            .unwrap()
            .duplicate
    );
    let raw = rusqlite::Connection::open(&path).unwrap();
    for table in [
        "embedding",
        "evidence",
        "entity",
        "predicate",
        "relationship",
    ] {
        raw.execute(&format!("DELETE FROM {table}"), []).unwrap();
        assert!(
            !learn(&store, &tok, &Fake("b"), text)
                .await
                .unwrap()
                .duplicate,
            "repair {table}"
        );
        assert!(
            learn(&store, &tok, &Fake("b"), text)
                .await
                .unwrap()
                .duplicate
        );
        assert_eq!(
            store
                .load_graph()
                .await
                .unwrap()
                .relationships()
                .next()
                .unwrap()
                .1
                .evidence
                .len(),
            1
        );
    }
}
#[tokio::test]
async fn top_k_is_filled_with_live_chunks() {
    let store = SqliteStore::open_mem().await.unwrap();
    let tok = tok();
    for text in ["Rust uses Tokio.", "Python provides tools."] {
        learn(&store, &tok, &Fake("a"), text).await.unwrap();
    }
    store
        .delete_chunk(&format!("{}:0", doc_id_for("Rust uses Tokio.")))
        .await
        .unwrap();
    let hits = search(&store, &Fake("a"), "Rust", 1).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].text, "Python provides tools.");
    assert!(
        search(&store, &Fake("a"), "Rust", 0)
            .await
            .unwrap()
            .is_empty()
    );
    store.delete_chunk(&hits[0].chunk_id).await.unwrap();
    assert!(
        search(&store, &Fake("a"), "Rust", 3)
            .await
            .unwrap()
            .is_empty()
    );
}
#[tokio::test]
async fn grouped_metadata_keeps_first_seen_order_labels_and_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("atlas.db");
    {
        let store = SqliteStore::open_durable(&path).await.unwrap();
        for (i, names) in [["Rust", "Tokio"], ["Cargo", "RUST"]].iter().enumerate() {
            let mut graph = KnowledgeGraph::new();
            let a = graph.intern_entity(names[0]);
            let b = graph.intern_entity(names[1]);
            let p = graph.intern_predicate(if i == 0 { "uses" } else { "extends" });
            if i == 0 {
                graph.set_kind(a, "language");
            }
            graph.link(
                a,
                p,
                b,
                EdgeOrigin::Explicit,
                Evidence {
                    doc_id: format!("{i}:0"),
                    sentence: "s".into(),
                },
            );
            store
                .apply_learn_write(&LearnWrite {
                    document: &DocumentRecord {
                        doc_id: i.to_string(),
                        text: "s".into(),
                    },
                    chunks: &[],
                    embeddings: &[],
                    graph: &graph,
                })
                .await
                .unwrap();
        }
    }
    let store = SqliteStore::open_durable(&path).await.unwrap();
    let graph = store.load_graph().await.unwrap();
    assert_eq!(
        graph
            .entities()
            .map(|(_, e)| e.display.as_str())
            .collect::<Vec<_>>(),
        ["Rust", "Tokio", "Cargo"]
    );
    assert_eq!(graph.entity(0).unwrap().kind.as_deref(), Some("language"));
    assert_eq!(
        graph
            .relationships()
            .map(|(_, r)| graph.entity_key(r.src).unwrap())
            .collect::<Vec<_>>(),
        ["rust", "cargo"]
    );
}

struct Invalid {
    vector: Vec<f32>,
}
impl Embedder for Invalid {
    fn model_id(&self) -> &str {
        "a"
    }
    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| self.vector.clone()).collect())
    }
}
#[tokio::test]
async fn invalid_vectors_are_rejected_before_storage() {
    let tok = tok();
    for vector in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
        let store = SqliteStore::open_mem().await.unwrap();
        assert!(
            learn(&store, &tok, &Invalid { vector }, "Rust uses Tokio.")
                .await
                .is_err()
        );
        assert!(
            !store
                .document_exists(&doc_id_for("Rust uses Tokio."))
                .await
                .unwrap()
        );
    }
}
#[tokio::test]
async fn invalid_queries_return_errors() {
    let store = SqliteStore::open_mem().await.unwrap();
    learn(&store, &tok(), &Fake("a"), "Rust uses Tokio.")
        .await
        .unwrap();
    for vector in [vec![], vec![1.0], vec![f32::NAN, 0.0]] {
        assert!(
            search(&store, &Invalid { vector }, "query", 1)
                .await
                .is_err()
        );
    }
}

struct Limited;
impl Embedder for Limited {
    fn model_id(&self) -> &str {
        "a"
    }
    fn validate_documents(&self, _: &[&str]) -> anyhow::Result<()> {
        anyhow::bail!("use shorter passages")
    }
    fn embed_documents(&self, _: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        panic!("limit must be checked first")
    }
}
#[tokio::test]
async fn limits_precede_writes_and_duplicate_detection() {
    let store = SqliteStore::open_mem().await.unwrap();
    let tok = tok();
    let text = "Rust uses Tokio.";
    assert!(learn(&store, &tok, &Limited, text).await.is_err());
    assert!(!store.document_exists(&doc_id_for(text)).await.unwrap());
    learn(&store, &tok, &Fake("a"), text).await.unwrap();
    assert!(learn(&store, &tok, &Limited, text).await.is_err());
    assert_eq!(store.load_embeddings("a").await.unwrap().len(), 1);
}

#[tokio::test]
async fn changed_chunk_layout_cannot_replace_old_model_text() {
    let store = SqliteStore::open_mem().await.unwrap();
    let document = DocumentRecord {
        doc_id: doc_id_for("full text"),
        text: "full text".into(),
    };
    let old = atlas_storage::ChunkRecord {
        chunk_id: format!("{}:0", document.doc_id),
        doc_id: document.doc_id.clone(),
        ordinal: 0,
        text: "full ".into(),
    };
    let old_vector = atlas_storage::EmbeddingRecord {
        chunk_id: old.chunk_id.clone(),
        model_id: "old".into(),
        vector: vec![1.0],
    };
    let graph = KnowledgeGraph::new();
    store
        .apply_learn_write(&LearnWrite {
            document: &document,
            chunks: std::slice::from_ref(&old),
            embeddings: &[old_vector],
            graph: &graph,
        })
        .await
        .unwrap();
    let mut changed = old.clone();
    changed.text = "full text".into();
    assert!(
        store
            .apply_learn_write(&LearnWrite {
                document: &document,
                chunks: &[changed],
                embeddings: &[],
                graph: &graph
            })
            .await
            .is_err()
    );
    assert_eq!(
        store.get_chunk(&old.chunk_id).await.unwrap().unwrap().text,
        old.text
    );
    assert_eq!(store.load_embeddings("old").await.unwrap().len(), 1);
}
