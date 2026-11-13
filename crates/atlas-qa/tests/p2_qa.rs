use atlas_embedding::Embedder;
use atlas_qa::{Answer, AnswerSentence, Confidence, ask};
use atlas_storage::{Backend, ChunkRecord, EmbeddingRecord, SqliteStore};
struct Fake;
impl Embedder for Fake {
    fn model_id(&self) -> &str {
        "qa-p2"
    }
    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
    }
}
async fn chunk(store: &SqliteStore, id: &str, text: &str, vector: Vec<f32>) {
    store
        .upsert_chunk(&ChunkRecord {
            chunk_id: id.into(),
            doc_id: "document".into(),
            ordinal: 0,
            text: text.into(),
        })
        .await
        .unwrap();
    store
        .upsert_embedding(&EmbeddingRecord {
            chunk_id: id.into(),
            model_id: "qa-p2".into(),
            vector,
        })
        .await
        .unwrap();
}
#[tokio::test]
async fn punctuation_and_function_words() {
    let store = SqliteStore::open_mem().await.unwrap();
    chunk(&store, "a", "Rust uses Tokio.", vec![1.0, 0.0]).await;
    assert!(!ask(&store, &Fake, "Tokio?").await.unwrap().insufficient);
    chunk(&store, "b", "Atlas is a learning engine.", vec![1.0, 0.0]).await;
    assert!(
        ask(&store, &Fake, "What is a quasar?")
            .await
            .unwrap()
            .insufficient
    );
}
#[tokio::test]
async fn repeated_terms_do_not_raise_confidence() {
    let store = SqliteStore::open_mem().await.unwrap();
    chunk(&store, "a", "Rust Rust Rust.", vec![1.0, 0.0]).await;
    assert_eq!(
        ask(&store, &Fake, "Rust").await.unwrap().confidence,
        Confidence::Medium
    );
}
#[tokio::test]
async fn confidence_uses_selected_sentence_score() {
    let store = SqliteStore::open_mem().await.unwrap();
    chunk(&store, "a", "unrelated text.", vec![1.0, 0.0]).await;
    chunk(&store, "b", "beta delta.", vec![0.1, 1.0]).await;
    assert_eq!(
        ask(&store, &Fake, "beta delta").await.unwrap().confidence,
        Confidence::Medium
    );
}
#[test]
fn rendering_keeps_complete_chunk_ids() {
    let id = "a".repeat(64);
    let answer = Answer {
        question: "q".into(),
        lead_in: None,
        confidence: Confidence::Medium,
        insufficient: false,
        multi_source: false,
        sentences: (0..2)
            .map(|i| AnswerSentence {
                text: "s".into(),
                doc_id: id.clone(),
                chunk_id: format!("{id}:{i}"),
                score: None,
            })
            .collect(),
    };
    let rendered = answer.render();
    assert!(rendered.contains(&format!("chunk {id}:0")));
    assert!(rendered.contains(&format!("chunk {id}:1")));
}
