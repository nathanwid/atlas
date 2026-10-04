//! Tests Atlas's own BPE trainer on the committed plain-English corpus.
//!
//! This corpus is used only for training. The cl100k ranks fixture is only
//! used for the compatibility diff test; it is never a training input.

mod common;

use common::{load_corpus, sample_texts};
use std::collections::HashSet;

const TARGET_VOCAB: usize = 512;

#[test]
fn training_is_deterministic() {
    let corpus = load_corpus();
    let first = atlas_tokenizer::train(&corpus, TARGET_VOCAB);
    let second = atlas_tokenizer::train(&corpus, TARGET_VOCAB);
    assert_eq!(first.ranks(), second.ranks());
}

#[test]
fn trained_tokenizer_encodes_corpus_within_vocab_and_compresses() {
    let corpus = load_corpus();
    let tok = atlas_tokenizer::train(&corpus, TARGET_VOCAB);
    assert_eq!(tok.vocab_size(), TARGET_VOCAB);

    let ids = tok.encode(&corpus);
    assert!(
        ids.iter().all(|&r| (r as usize) < TARGET_VOCAB),
        "encoding must only produce learned ranks"
    );
    // A trained table on the text it was trained on must compress: no merge
    // would ever be chosen if it did not reduce token count where it fires.
    assert!(
        ids.len() * 2 < corpus.len(),
        "expected at least 2x compression, got {} tokens for {} bytes",
        ids.len(),
        corpus.len()
    );
    assert_eq!(
        tok.decode(&ids).unwrap(),
        corpus,
        "trained tokenizer must roundtrip"
    );
}

#[test]
fn trained_tokenizer_encodes_unseen_text_within_vocab() {
    let corpus = load_corpus();
    let tok = atlas_tokenizer::train(&corpus, TARGET_VOCAB);
    for text in sample_texts() {
        let ids = tok.encode(text);
        assert!(ids.iter().all(|&r| (r as usize) < TARGET_VOCAB));
        assert_eq!(tok.decode(&ids).unwrap(), text);
    }
}

#[test]
fn single_bytes_are_ranks_zero_to_255() {
    let corpus = load_corpus();
    let tok = atlas_tokenizer::train(&corpus, TARGET_VOCAB);
    let ranks = tok.ranks();
    for b in 0u8..=255 {
        assert_eq!(ranks.get(&[b][..]), Some(&(b as atlas_tokenizer::Rank)));
    }
}

#[test]
fn vocab_holds_no_duplicate_byte_strings() {
    let corpus = load_corpus();
    let tok = atlas_tokenizer::train(&corpus, TARGET_VOCAB);
    let unique: HashSet<&[u8]> = tok.ranks().keys().map(|b| &b[..]).collect();
    assert_eq!(unique.len(), tok.vocab_size());
}
