use atlas_embedding::{Embedder, LearnedEmbedder, SkipGramModel, TrainerConfig, cache_key, train};
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
fn trained(seed: u64) -> (SkipGramModel, Vocab) {
    let mut vocab = Vocab::default();
    let model = train(
        &mut vocab,
        &tok(),
        "alpha beta alpha gamma",
        &TrainerConfig {
            dim: 4,
            epochs: 2,
            seed,
            ..Default::default()
        },
    );
    (model, vocab)
}
#[test]
fn reordered_vocabulary_is_rejected() {
    let (model, vocab) = trained(1);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model");
    model.save(&path).unwrap();
    let mut other = Vocab::default();
    for i in (0..vocab.len() as u32).rev() {
        other.get_or_insert(vocab.token(i).unwrap());
    }
    assert!(SkipGramModel::load(&path, &other).is_err());
    assert!(std::panic::catch_unwind(|| LearnedEmbedder::new(model, tok(), other)).is_err());
}
#[test]
fn identity_tracks_weights_and_survives_reload() {
    let (model, vocab) = trained(1);
    let (other, other_vocab) = trained(2);
    let first = LearnedEmbedder::new(model.clone(), tok(), vocab.clone());
    let second = LearnedEmbedder::new(other, tok(), other_vocab);
    assert_ne!(first.model_id(), second.model_id());
    assert_ne!(
        cache_key(first.model_id(), "q"),
        cache_key(second.model_id(), "q")
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model");
    model.save(&path).unwrap();
    let loaded = LearnedEmbedder::new(SkipGramModel::load(&path, &vocab).unwrap(), tok(), vocab);
    assert_eq!(first.model_id(), loaded.model_id());
}
#[test]
fn legacy_files_are_preserved() {
    #[derive(serde::Serialize)]
    struct Legacy {
        version: u32,
        dim: u32,
        vocab_len: u32,
        w_in: Vec<f32>,
        w_out: Vec<f32>,
    }
    let (model, _) = trained(1);
    let mut vocab = Vocab::default();
    vocab.get_or_insert(b"a");
    let bytes = bincode::serialize(&Legacy {
        version: 1,
        dim: 1,
        vocab_len: 1,
        w_in: vec![1.0],
        w_out: vec![0.0],
    })
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy");
    std::fs::write(&path, &bytes).unwrap();
    assert!(SkipGramModel::load(&path, &vocab).is_err());
    assert!(model.save(&path).is_err());
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}

#[test]
fn changed_tokenizer_is_rejected_before_inference_or_learning() {
    let (mut model, mut vocab) = trained(1);
    let byte_tok = BpeTokenizer::from_ranks(
        (0..256u32)
            .map(|b| (vec![b as u8].into_boxed_slice(), b))
            .collect(),
    );
    assert!(
        std::panic::catch_unwind(|| LearnedEmbedder::new(
            model.clone(),
            byte_tok.clone(),
            vocab.clone()
        ))
        .is_err()
    );
    let before = vocab.len();
    assert!(
        model
            .learn(
                &mut vocab,
                &byte_tok,
                "new text",
                &TrainerConfig {
                    dim: 4,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert_eq!(vocab.len(), before);
}
