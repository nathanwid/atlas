//! Real-model integration test for the chunk + embed pipeline (M04
//! groundwork).
//!
//! Ignored by default: the first run downloads ~100 MB of ONNX weights into
//! fastembed's cache directory and needs network access. Run explicitly with
//! `cargo test -p atlas-embedding -- --ignored`.

use atlas_embedding::{Chunker, FastembedEmbedder, VectorCache, embed_chunks};

fn fixture(name: &str) -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/"
    )
    .to_string()
        + name;
    std::fs::read_to_string(path).expect("shared fixture")
}

#[test]
#[ignore = "downloads ~100 MB model weights on first run; requires network"]
fn embeds_corpus_chunks_with_real_model() {
    let corpus = fixture("corpus.txt");
    let ranks = std::fs::File::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
    ))
    .expect("ranks fixture");
    let tokenizer =
        atlas_tokenizer::BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(ranks))
            .expect("ranks parse");

    let chunks = Chunker::new(&tokenizer, 128, 0).chunk(&corpus);
    assert!(chunks.len() > 5, "corpus should produce many chunks");

    let embedder = FastembedEmbedder::bge_small_en_v1_5().expect("model loads");
    let dir = tempfile::tempdir().unwrap();
    let mut cache = VectorCache::load(dir.path().join("vectors.bin"));

    let texts: Vec<&str> = chunks.iter().map(|c| c.text).collect();
    let vectors = embed_chunks(&embedder, &mut cache, &texts).expect("embeds");

    assert_eq!(vectors.len(), chunks.len());
    for v in &vectors {
        assert_eq!(
            v.len(),
            384,
            "bge-small-en-v1.5 produces 384-dimensional vectors"
        );
        assert!(v.iter().all(|x| x.is_finite()));
    }
    assert_eq!(cache.len(), chunks.len(), "every chunk vector is cached");

    // Second pass over the same chunks: pure cache hits, identical vectors.
    let again = embed_chunks(&embedder, &mut cache, &texts).expect("cache hits");
    assert_eq!(again, vectors);
}
