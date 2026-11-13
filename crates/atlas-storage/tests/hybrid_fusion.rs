//! Fixed ranks distinguish fusion from a union of disjoint result lists.

use atlas_embedding::Embedder;
use atlas_graph::{EdgeOrigin, Evidence, KnowledgeGraph};
use atlas_storage::{
    Backend, ChunkRecord, EmbeddingRecord, ResultSource, SearchMode, SqliteStore, hybrid_search,
};

struct QueryEmbedder;

impl Embedder for QueryEmbedder {
    fn model_id(&self) -> &str {
        "fusion-test"
    }
    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
    }
}

async fn fixture(with_graph: bool) -> SqliteStore {
    let store = SqliteStore::open_mem().await.unwrap();
    for (id, vector) in [
        ("a", vec![1.0, 0.0]),
        ("b", vec![0.8, 0.6]),
        ("c", vec![0.0, 1.0]),
    ] {
        store.upsert_document(id, id).await.unwrap();
        let chunk_id = format!("{id}:0");
        store
            .upsert_chunk(&ChunkRecord {
                chunk_id: chunk_id.clone(),
                doc_id: id.into(),
                ordinal: 0,
                text: id.into(),
            })
            .await
            .unwrap();
        store
            .upsert_embedding(&EmbeddingRecord {
                chunk_id,
                model_id: "fusion-test".into(),
                vector,
            })
            .await
            .unwrap();
    }
    if with_graph {
        let mut graph = KnowledgeGraph::new();
        let rust = graph.intern_entity("Rust");
        let tokio = graph.intern_entity("Tokio");
        let uses = graph.intern_predicate("uses");
        for id in ["b", "c"] {
            graph.link(
                rust,
                uses,
                tokio,
                EdgeOrigin::Explicit,
                Evidence {
                    doc_id: format!("{id}:0"),
                    sentence: id.into(),
                },
            );
        }
        store.persist_graph(&graph).await.unwrap();
    }
    store
}

#[tokio::test]
async fn shared_hit_receives_both_contributions_and_overtakes_vector_leader() {
    let store = fixture(true).await;
    let config = Default::default();
    let vector = hybrid_search(
        &store,
        &QueryEmbedder,
        "query",
        2,
        &config,
        SearchMode::Vector,
    )
    .await
    .unwrap();
    assert_eq!(
        vector
            .iter()
            .map(|hit| hit.chunk_id.as_str())
            .collect::<Vec<_>>(),
        ["a:0", "b:0"]
    );
    let hybrid = hybrid_search(
        &store,
        &QueryEmbedder,
        "query",
        2,
        &config,
        SearchMode::Hybrid,
    )
    .await
    .unwrap();
    // b has vector rank 2 and graph rank 1: 1/61 + 1/60 > a's 1/60.
    assert_eq!(
        hybrid
            .iter()
            .map(|hit| hit.chunk_id.as_str())
            .collect::<Vec<_>>(),
        ["b:0", "a:0"]
    );
    assert_eq!(hybrid[0].source, ResultSource::Both);
    assert_eq!(hybrid[0].score, vector[1].score);
    assert_eq!(
        hybrid[0].via,
        [
            "vector rank 2 (score 0.8000)",
            "graph rank 1 via Rust -[uses]-> Tokio"
        ]
    );
    assert_eq!(hybrid[1].source, ResultSource::Vector);
    assert_eq!(
        hybrid,
        hybrid_search(
            &store,
            &QueryEmbedder,
            "query",
            2,
            &config,
            SearchMode::Hybrid
        )
        .await
        .unwrap()
    );

    let graph = hybrid_search(
        &store,
        &QueryEmbedder,
        "query",
        2,
        &config,
        SearchMode::Graph,
    )
    .await
    .unwrap();
    assert_eq!(graph.len(), 1);
    assert_eq!(graph[0].chunk_id, "c:0");
    assert_eq!(graph[0].source, ResultSource::Graph);
    assert_eq!(graph[0].score, None);
    assert_eq!(graph[0].via, ["graph rank 1 via Rust -[uses]-> Tokio"]);
    store.delete_chunk("c:0").await.unwrap();
    assert!(
        hybrid_search(
            &store,
            &QueryEmbedder,
            "query",
            2,
            &config,
            SearchMode::Graph
        )
        .await
        .unwrap()
        .is_empty()
    );
}

#[tokio::test]
async fn empty_graph_keeps_vector_results() {
    let store = fixture(false).await;
    let config = Default::default();
    let vector = hybrid_search(
        &store,
        &QueryEmbedder,
        "query",
        2,
        &config,
        SearchMode::Vector,
    )
    .await
    .unwrap();
    let hybrid = hybrid_search(
        &store,
        &QueryEmbedder,
        "query",
        2,
        &config,
        SearchMode::Hybrid,
    )
    .await
    .unwrap();
    assert_eq!(hybrid, vector);
}
