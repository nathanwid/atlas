//! QA must tolerate evidence referencing chunks that no longer exist
//! (hardening Phase 1): retrieval skips missing chunks during expansion,
//! and answers cite only chunks that are still recoverable.

use atlas_embedding::Embedder;
use atlas_qa::ask;
use atlas_storage::{Store, learn};
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
    let store = Store::open_mem().await.unwrap();
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
    let store = Store::open_mem().await.unwrap();
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
