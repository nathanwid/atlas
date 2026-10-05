//! M05 corpus training evaluation.
//!
//! Trains the Skip-Gram model on the committed corpus with the cl100k
//! tokenizer and the M03 vocabulary, then prints an honest report: vocab
//! size, timing, and nearest-neighbor probes. Assertions are mechanical
//! only (dimensions, finiteness, roundtrip, incremental growth); neighbor
//! quality is report-only.

use atlas_embedding::{Embedder, LearnedEmbedder, SkipGramModel, TrainerConfig, train};
use atlas_tokenizer::BpeTokenizer;
use atlas_vocab::Vocab;
use std::time::Instant;

fn fixture(name: &str) -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/"
    )
    .to_string()
        + name
}

#[test]
fn trains_on_corpus_and_supports_incremental_learning() {
    let corpus = std::fs::read_to_string(fixture("corpus.txt")).expect("corpus fixture");
    let ranks = std::fs::File::open(fixture("cl100k.tiktoken")).expect("ranks fixture");
    let tokenizer =
        BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(ranks)).expect("ranks parse");

    let config = TrainerConfig::default();
    let mut vocab = Vocab::default();

    let start = Instant::now();
    let model = train(&mut vocab, &tokenizer, &corpus, &config);
    let elapsed = start.elapsed();

    // Mechanical assertions.
    assert!(
        vocab.len() > 500,
        "real corpus must produce a real vocabulary"
    );
    assert_eq!(model.vocab_len(), vocab.len());
    assert_eq!(model.dim(), config.dim);
    assert!(model.token_vector(0).unwrap().iter().all(|x| x.is_finite()));

    println!(
        "trained: vocab={} dim={} corpus={}B in {:.2?} (config: {:?})",
        vocab.len(),
        config.dim,
        corpus.len(),
        elapsed,
        config
    );

    // Report-only quality probes: nearest neighbors for a few corpus words.
    for word in ["tokenizer", "search", "graph", "vectors"] {
        let Some(id) = vocab.get(word.as_bytes()) else {
            println!("probe {word:?}: not a standalone token in this vocabulary");
            continue;
        };
        let neighbors = model.nearest(id, 5).unwrap();
        let rendered: Vec<String> = neighbors
            .iter()
            .map(|(id, score)| {
                (
                    std::str::from_utf8(vocab.token(*id).unwrap())
                        .unwrap_or("<bytes>")
                        .to_string(),
                    *score,
                )
            })
            .map(|(t, s)| format!("{t:?}={s:.3}"))
            .collect();
        println!("probe {word:?}: {}", rendered.join(", "));
    }

    // Persistence roundtrip against the same vocabulary.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("skipgram.bin");
    model.save(&path).unwrap();
    let reloaded = SkipGramModel::load(&path, &vocab).expect("reload");
    for id in 0..vocab.len() as u32 {
        assert_eq!(reloaded.token_vector(id), model.token_vector(id));
    }

    // Loading against a different vocabulary must fail loudly.
    let mut other = Vocab::default();
    other.get_or_insert(b"unrelated");
    assert!(SkipGramModel::load(&path, &other).is_err());

    // Incremental learning: new text grows the vocabulary and the model,
    // and the text becomes embeddable afterwards.
    let before = vocab.len();
    let mut model = model;
    model
        .learn(
            &mut vocab,
            &tokenizer,
            "Atlas learned about knowledge graphs today.",
            &config,
        )
        .expect("incremental learn");
    assert!(vocab.len() > before, "new text must grow the vocabulary");
    assert_eq!(model.vocab_len(), vocab.len());

    let embedder = LearnedEmbedder::new(model, tokenizer, vocab);
    let v = embedder
        .embed_documents(&["knowledge graphs"])
        .unwrap()
        .remove(0);
    assert_eq!(v.len(), config.dim);
    assert_eq!(
        embedder.model_id(),
        format!(
            "atlas-skipgram-d{}-v{}",
            config.dim,
            embedder.model().vocab_len()
        )
    );
}
