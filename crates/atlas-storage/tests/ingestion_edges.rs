//! Ingestion edge cases for the durable learn flow (hardening Phase 1).
//!
//! Asserts current, intended behavior for: empty text, whitespace-only
//! text, oversized single pre-tokens (hard-split path), unicode-heavy
//! input, and re-ingestion after a simulated partial failure.
//!
//! Semantics pinned here (do not change without a plan):
//! * Empty text stores a document with zero chunks — valid, learnable.
//! * Whitespace-only text stores one whitespace chunk — byte-level tokens
//!   make whitespace content, not noise.
//! * Every learn is repairable: a document row without its complete chunk
//!   set is completed by the next learn call.

use atlas_embedding::Embedder;
use atlas_storage::{ChunkRecord, Store, doc_id_for, learn};
use atlas_tokenizer::BpeTokenizer;

/// Deterministic fake embedder (same shape as the M7 tests): vector =
/// [text length, starts with 'a'].
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

async fn learned(text: &str) -> (Store, atlas_storage::LearnSummary) {
    let store = Store::open_mem().await.unwrap();
    let summary = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    (store, summary)
}

#[tokio::test]
async fn empty_text_stores_a_chunkless_document() {
    let (store, summary) = learned("").await;
    assert!(!summary.duplicate);
    assert_eq!(summary.chunks, 0);
    let doc_id = doc_id_for("");
    assert!(store.document_exists(&doc_id).await.unwrap());
    assert!(
        store
            .load_embeddings("fake-test-model")
            .await
            .unwrap()
            .is_empty()
    );
    // Re-learning empty text is a clean duplicate (complete = zero chunks).
    let again = learn(&store, &tokenizer(), &FakeEmbedder, "")
        .await
        .unwrap();
    assert!(again.duplicate);
}

#[tokio::test]
async fn whitespace_only_text_stores_one_whitespace_chunk() {
    let text = "   ";
    let (store, summary) = learned(text).await;
    assert_eq!(
        summary.chunks, 1,
        "whitespace is content for a byte-level tokenizer"
    );
    let doc_id = doc_id_for(text);
    let chunk: ChunkRecord = store
        .get_chunk(&format!("{doc_id}:0"))
        .await
        .unwrap()
        .expect("chunk exists");
    assert_eq!(chunk.text, text);
}

#[tokio::test]
async fn oversized_single_piece_hard_splits() {
    // 10k identical letters form ONE pre-token (letter run), forcing the
    // chunker's hard-split path across many chunks.
    let text = "x".repeat(10_000);
    let (store, summary) = learned(&text).await;
    assert!(
        summary.chunks > 1,
        "10k chars must exceed one 128-token chunk"
    );

    // Chunks partition the input exactly, in order.
    let doc_id = doc_id_for(&text);
    let mut joined = String::new();
    for ordinal in 0..summary.chunks {
        let chunk: ChunkRecord = store
            .get_chunk(&format!("{doc_id}:{ordinal}"))
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("chunk {ordinal} missing"));
        joined.push_str(&chunk.text);
    }
    assert_eq!(joined, text, "hard-split chunks must reconstruct the input");
}

#[tokio::test]
async fn unicode_heavy_text_roundtrips_through_storage() {
    let text = "Rust 🦀と日本語のテキスト — naïve café. 日本語の文章が続きます。 Ünïcödé strings handle mixing. "
        .repeat(8);
    let (store, summary) = learned(&text).await;
    assert!(summary.chunks >= 1);

    let doc_id = doc_id_for(&text);
    let mut joined = String::new();
    for ordinal in 0..summary.chunks {
        let chunk: ChunkRecord = store
            .get_chunk(&format!("{doc_id}:{ordinal}"))
            .await
            .unwrap()
            .expect("chunk exists");
        joined.push_str(&chunk.text);
    }
    assert_eq!(joined, text, "unicode text must roundtrip byte-exactly");

    // Graph extraction ran without choking on non-latin scripts.
    let _ = store.load_graph().await.unwrap();
}

#[tokio::test]
async fn reingestion_repairs_a_simulated_partial_failure() {
    let store = Store::open_mem().await.unwrap();
    let text = "alpha beta gamma delta epsilon zeta eta theta";
    let doc_id = doc_id_for(text);

    // Simulate a partial failure: the document row was written, but the
    // process died before any chunk/embedding row.
    store.upsert_document(&doc_id, text).await.unwrap();
    assert!(
        store
            .get_chunk(&format!("{doc_id}:0"))
            .await
            .unwrap()
            .is_none()
    );

    // The next learn must repair, not short-circuit as a duplicate.
    let summary = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    assert!(
        !summary.duplicate,
        "partial state is not a healthy duplicate"
    );
    assert_eq!(summary.chunks, 1);
    assert!(
        store
            .get_chunk(&format!("{doc_id}:0"))
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        store
            .load_embeddings("fake-test-model")
            .await
            .unwrap()
            .len(),
        1
    );

    // And the repaired document is a clean duplicate afterwards.
    let again = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    assert!(again.duplicate);
}

#[tokio::test]
async fn reingestion_repairs_missing_later_chunks() {
    let store = Store::open_mem().await.unwrap();
    // Long enough to produce several chunks.
    let text = "filler ".repeat(600);
    let doc_id = doc_id_for(&text);

    let first = learn(&store, &tokenizer(), &FakeEmbedder, &text)
        .await
        .unwrap();
    assert!(first.chunks > 2);

    // Simulate losing a late chunk (e.g. torn write): delete it directly.
    store
        .delete_chunk(&format!("{doc_id}:{}", first.chunks - 1))
        .await
        .unwrap();

    let summary = learn(&store, &tokenizer(), &FakeEmbedder, &text)
        .await
        .unwrap();
    assert!(!summary.duplicate, "missing chunk must trigger repair");
    for ordinal in 0..summary.chunks {
        assert!(
            store
                .get_chunk(&format!("{doc_id}:{ordinal}"))
                .await
                .unwrap()
                .is_some(),
            "chunk {ordinal} must exist after repair"
        );
    }
}
