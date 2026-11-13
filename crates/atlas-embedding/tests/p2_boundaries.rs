use atlas_embedding::{Chunker, Embedder, TrainerConfig, train};
use atlas_tokenizer::BpeTokenizer;
use atlas_vocab::Vocab;
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
#[test]
fn overlap_stops_at_eof() {
    let tok = tok();
    let chunks = Chunker::new(&tok, 20, 2).chunk("a b c");
    assert_eq!(chunks.len(), 1);
}
#[test]
fn invalid_training_keeps_vocabulary_unchanged() {
    let tok = tok();
    let mut vocab = Vocab::default();
    let config = TrainerConfig {
        dim: 0,
        ..Default::default()
    };
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        train(&mut vocab, &tok, "alpha beta", &config)
    }));
    assert!(vocab.is_empty());
}
#[test]
fn invalid_incremental_training_is_rejected_before_mutation() {
    let tok = tok();
    let mut vocab = Vocab::default();
    let config = TrainerConfig {
        dim: 4,
        epochs: 1,
        ..Default::default()
    };
    let mut model = train(&mut vocab, &tok, "alpha beta", &config);
    let before = vocab.len();
    let invalid = TrainerConfig {
        lr: f32::NAN,
        ..config
    };
    assert!(
        model
            .learn(&mut vocab, &tok, "gamma delta", &invalid)
            .is_err()
    );
    assert_eq!(vocab.len(), before);
}
struct Empty;
impl Embedder for Empty {
    fn model_id(&self) -> &str {
        "empty"
    }
    fn embed_documents(&self, _: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(vec![])
    }
}
#[test]
fn empty_query_response_is_an_error() {
    assert!(Empty.embed_query("q").is_err());
}
