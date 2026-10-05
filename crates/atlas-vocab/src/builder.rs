//! Deterministic vocabulary construction from tokenized text.
//!
//! IDs are assigned by first occurrence: walking the corpus piece by piece
//! (pre-tokenizer order), token by token (encoder order), inserting each
//! unseen token with `get_or_insert`. The result depends only on the corpus
//! and the tokenizer — no hash-map iteration is involved — so the same
//! inputs always produce the same IDs.

use crate::vocab::Vocab;
use atlas_tokenizer::BpeTokenizer;
use atlas_tokenizer::pretokenize::pretokenize;

/// Build a vocabulary over `corpus` using `tokenizer`.
pub fn build(tokenizer: &BpeTokenizer, corpus: &str) -> Vocab {
    let mut vocab = Vocab::default();
    for piece in pretokenize(corpus) {
        for rank in tokenizer.encode(piece) {
            let bytes = tokenizer
                .token_bytes(rank)
                .expect("encode only emits ranks from this tokenizer");
            vocab.get_or_insert(bytes);
        }
    }
    vocab
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> BpeTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
        );
        let f = std::fs::File::open(path).expect("ranks fixture");
        BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("parses")
    }

    fn corpus() -> String {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../atlas-tokenizer/tests/fixtures/corpus.txt"
        );
        std::fs::read_to_string(path).expect("corpus fixture")
    }

    #[test]
    fn build_is_deterministic() {
        let tok = tokenizer();
        let corpus = corpus();
        let a = build(&tok, &corpus);
        let b = build(&tok, &corpus);
        assert_eq!(a.len(), b.len());
        for id in 0..a.len() as u32 {
            assert_eq!(a.token(id), b.token(id), "id {id} differs between runs");
        }
    }

    #[test]
    fn ids_follow_first_occurrence_order() {
        let tok = tokenizer();
        // Pieces become tokens in encounter order, so every token of the
        // first pre-token sorts before every token of the second.
        let vocab = build(&tok, "zulu alpha");
        let first_rank = tok.encode("zulu")[0];
        let second_rank = tok.encode(" alpha")[0];
        let first = vocab.get(tok.token_bytes(first_rank).unwrap()).unwrap();
        let second = vocab.get(tok.token_bytes(second_rank).unwrap()).unwrap();
        assert!(
            first < second,
            "first occurrence must win: {first} vs {second}"
        );
    }

    #[test]
    fn corpus_tokens_all_resolve_and_ids_are_dense() {
        let tok = tokenizer();
        let corpus = corpus();
        let vocab = build(&tok, &corpus);
        assert!(
            vocab.len() > 100,
            "real corpus must produce a real vocabulary"
        );

        // Every token the tokenizer emits for the corpus is in the vocab.
        for piece in pretokenize(&corpus) {
            for rank in tok.encode(piece) {
                let bytes = tok.token_bytes(rank).unwrap();
                assert!(vocab.get(bytes).is_some());
            }
        }

        // Dense + bijective: ids 0..len each map to a unique token.
        let mut seen = std::collections::HashSet::new();
        for id in 0..vocab.len() as u32 {
            let token = vocab.token(id).expect("dense ids");
            assert!(seen.insert(token.to_vec()), "duplicate token at id {id}");
        }
        assert_eq!(seen.len(), vocab.len());
    }

    #[test]
    fn built_vocab_survives_persistence_and_still_resolves() {
        let tok = tokenizer();
        let corpus = corpus();
        let vocab = build(&tok, &corpus);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.bin");
        vocab.save(&path).unwrap();
        let loaded = crate::Vocab::load(&path).unwrap();

        assert_eq!(loaded.len(), vocab.len());
        let sample = pretokenize(&corpus)[0];
        for rank in tok.encode(sample) {
            let bytes = tok.token_bytes(rank).unwrap();
            assert_eq!(loaded.get(bytes), vocab.get(bytes));
        }
    }

    #[test]
    fn empty_corpus_builds_empty_vocab() {
        let tok = tokenizer();
        assert!(build(&tok, "").is_empty());
    }
}
