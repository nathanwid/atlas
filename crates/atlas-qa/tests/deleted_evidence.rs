//! QA must tolerate evidence referencing chunks that no longer exist
//! (hardening Phase 1): retrieval skips missing chunks during expansion,
//! and answers cite only chunks that are still recoverable.

use atlas_embedding::Embedder;
use atlas_qa::ask;
use atlas_storage::Backend;
use atlas_storage::{SqliteStore, learn};
use atlas_tokenizer::BpeTokenizer;
use std::collections::HashSet;

/// Deterministic word-overlap embedder (same pattern as the M8/M9 eval).
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

#[tokio::test]
async fn answers_cite_only_recoverable_chunks_after_deletion() {
    let store = SqliteStore::open_mem().await.unwrap();
    let tok = tokenizer();
    let embedder = WordOverlapEmbedder;

    // Two documents sharing the Rust entity: doc 1 (vector seed) links to
    // doc 0 through the graph, so doc 0 is a graph-expansion candidate.
    learn(
        &store,
        &tok,
        &embedder,
        "Rust uses Tokio for async runtimes.",
    )
    .await
    .unwrap();
    learn(
        &store,
        &tok,
        &embedder,
        "Rust requires careful tooling choices.",
    )
    .await
    .unwrap();

    let doc0 = format!(
        "{}:0",
        atlas_storage::doc_id_for("Rust uses Tokio for async runtimes.")
    );
    let doc1 = format!(
        "{}:0",
        atlas_storage::doc_id_for("Rust requires careful tooling choices.")
    );

    // Sanity before deletion: both chunks answerable.
    let before = ask(&store, &embedder, "Rust tooling").await.unwrap();
    assert!(!before.insufficient);
    assert!(
        before
            .lead_in
            .as_deref()
            .unwrap()
            .contains("Rust -[uses]-> Tokio")
    );
    let cited: HashSet<&str> = before
        .sentences
        .iter()
        .map(|s| s.chunk_id.as_str())
        .collect();
    assert!(cited.contains(doc1.as_str()), "doc 1 must be a vector hit");

    // Delete the doc-0 chunk: its evidence now references a missing chunk.
    store.delete_chunk(&doc0).await.unwrap();

    // The answer must not fail or cite the deleted chunk.
    let after = ask(&store, &embedder, "Rust tooling").await.unwrap();
    assert!(
        !after.insufficient,
        "doc 1 still exists and answers the question"
    );
    assert!(
        !after
            .lead_in
            .as_deref()
            .unwrap_or_default()
            .contains("Rust -[uses]-> Tokio")
    );
    assert!(!after.render().contains("Rust -[uses]-> Tokio"));
    for sentence in &after.sentences {
        assert_ne!(sentence.chunk_id, doc0, "deleted chunk must never be cited");
        let chunk = store
            .get_chunk(&sentence.chunk_id)
            .await
            .unwrap()
            .expect("every cited chunk must be recoverable");
        assert!(
            chunk.text.contains(&sentence.text),
            "citations must stay grounded after deletion"
        );
    }
}

#[tokio::test]
async fn deleting_all_evidence_leads_to_insufficient() {
    let store = SqliteStore::open_mem().await.unwrap();
    let tok = tokenizer();
    let embedder = WordOverlapEmbedder;

    let text = "alpha beta gamma delta epsilon";
    learn(&store, &tok, &embedder, text).await.unwrap();
    store
        .delete_chunk(&format!("{}:0", atlas_storage::doc_id_for(text)))
        .await
        .unwrap();

    // Word overlap still matches the question to the (deleted) chunk's
    // vector, but the chunk is gone: the answer must degrade honestly.
    let answer = ask(&store, &embedder, "alpha beta gamma").await.unwrap();
    assert!(
        answer.insufficient,
        "no recoverable evidence means insufficient"
    );
    assert!(answer.sentences.is_empty());
}

/// The answer chunk is retrieved. Supporting chunks have no embedding, so
/// lead-in validation must also handle evidence outside the retrieved hits.
async fn lead_in_store(src: &str, evidence: Vec<atlas_graph::Evidence>) -> SqliteStore {
    let store = SqliteStore::open_mem().await.unwrap();
    let text = format!("{src} tooling facts remain available.");
    put_chunk(&store, "answer:0", &text).await;
    store
        .upsert_embedding(&atlas_storage::EmbeddingRecord {
            chunk_id: "answer:0".into(),
            model_id: "word-overlap".into(),
            vector: WordOverlapEmbedder.embed_query(&text).unwrap(),
        })
        .await
        .unwrap();
    let mut graph = atlas_graph::KnowledgeGraph::new();
    let source = graph.intern_entity(src);
    let destination = graph.intern_entity("Tokio");
    let predicate = graph.intern_predicate("uses");
    for ev in evidence {
        graph.link(
            source,
            predicate,
            destination,
            atlas_graph::EdgeOrigin::Explicit,
            ev,
        );
    }
    store.persist_graph(&graph).await.unwrap();
    store
}

async fn put_chunk(store: &SqliteStore, id: &str, text: &str) {
    let doc_id = id.split(':').next().unwrap();
    store.upsert_document(doc_id, text).await.unwrap();
    store
        .upsert_chunk(&atlas_storage::ChunkRecord {
            chunk_id: id.into(),
            doc_id: doc_id.into(),
            ordinal: 0,
            text: text.into(),
        })
        .await
        .unwrap();
}

fn sighting(chunk_id: &str, sentence: &str) -> atlas_graph::Evidence {
    atlas_graph::Evidence {
        doc_id: chunk_id.into(),
        sentence: sentence.into(),
    }
}

#[tokio::test]
async fn lead_in_uses_first_live_source_and_rejects_stale_or_empty_evidence() {
    let sentence = "Rust uses Tokio";
    let store = lead_in_store(
        "Rust",
        vec![
            sighting("missing:0", sentence),
            sighting("wrong:0", sentence),
            sighting("empty:0", ""),
            sighting("live:0", sentence),
        ],
    )
    .await;
    put_chunk(&store, "wrong:0", "Different content").await;
    put_chunk(&store, "empty:0", "Different content").await;
    put_chunk(&store, "live:0", "Rust uses Tokio for async runtimes.").await;
    let answer = ask(&store, &WordOverlapEmbedder, "Rust tooling")
        .await
        .unwrap();
    let lead = answer.lead_in.unwrap();
    assert!(lead.contains("Rust -[uses]-> Tokio"));
    assert!(lead.contains("[doc live, chunk live:0]"), "{lead}");
    assert!(!lead.contains("missing:0") && !lead.contains("wrong:0") && !lead.contains("empty:0"));

    store.delete_chunk("live:0").await.unwrap();
    let answer = ask(&store, &WordOverlapEmbedder, "Rust tooling")
        .await
        .unwrap();
    assert!(!answer.insufficient);
    assert!(answer.lead_in.is_none());
    assert!(!answer.render().contains("Rust -[uses]-> Tokio"));
}

#[tokio::test]
async fn lead_in_matches_whole_entity_phrases() {
    let store = lead_in_store("Go", vec![sighting("live:0", "Go uses Tokio")]).await;
    put_chunk(&store, "live:0", "Go uses Tokio.").await;
    let answer = ask(&store, &WordOverlapEmbedder, "Cargo tooling")
        .await
        .unwrap();
    assert!(!answer.insufficient);
    assert!(answer.lead_in.is_none(), "Go must not match Cargo");
    for question in ["go tooling", "(GO) tooling"] {
        assert!(
            ask(&store, &WordOverlapEmbedder, question)
                .await
                .unwrap()
                .lead_in
                .is_some()
        );
    }
    let store = lead_in_store("New York", vec![sighting("live:0", "New York uses Tokio")]).await;
    put_chunk(&store, "live:0", "New York uses Tokio.").await;
    assert!(
        ask(&store, &WordOverlapEmbedder, "(NEW YORK) tooling")
            .await
            .unwrap()
            .lead_in
            .is_some()
    );
    assert!(
        ask(&store, &WordOverlapEmbedder, "new yorker tooling")
            .await
            .unwrap()
            .lead_in
            .is_none()
    );
}
