//! Diff test: Atlas's byte-level BPE must produce token-for-token identical
//! token IDs to tiktoken-rs cl100k_base, which is the reference
//! implementation for this vocabulary. tiktoken-rs 0.12 already returns
//! `Vec<u32>`, matching Atlas's `Rank`.

mod common;

use common::{build_hf_cl100k, load_atlas_bpe, load_corpus, load_ranks, sample_texts};
use tiktoken_rs::cl100k_base;

#[test]
fn atlas_matches_tiktoken_on_samples() {
    let oracle = cl100k_base().expect("oracle loads");
    let atlas = load_atlas_bpe();
    for text in sample_texts() {
        let expected = oracle.encode_ordinary(text);
        assert_eq!(
            atlas.encode(text),
            expected,
            "heap engine mismatch for {text:?}"
        );
        assert_eq!(
            atlas.encode_naive(text),
            expected,
            "naive engine mismatch for {text:?}"
        );
    }
}

#[test]
fn atlas_matches_tiktoken_on_corpus() {
    let oracle = cl100k_base().expect("oracle loads");
    let atlas = load_atlas_bpe();
    let corpus = load_corpus();
    let expected = oracle.encode_ordinary(&corpus);
    assert_eq!(atlas.encode(&corpus), expected);
    assert_eq!(atlas.encode_naive(&corpus), expected);
    // The corpus is real prose, so the published vocabulary should compress
    // it substantially below one token per byte.
    assert!(
        expected.len() < corpus.len(),
        "expected compression, got {} tokens for {} bytes",
        expected.len(),
        corpus.len()
    );
}

#[test]
fn encode_decode_roundtrips_exactly() {
    let atlas = load_atlas_bpe();
    let corpus = load_corpus();
    let mut texts: Vec<&str> = sample_texts();
    texts.push(&corpus);
    for text in texts {
        assert_eq!(atlas.decode(&atlas.encode(text)).unwrap(), text);
    }
}

/// Validates the benchmark setup: the HF tokenizer we derive from the ranks
/// file (vocab via byte-to-unicode, merges reconstructed from rank order)
/// must also match the oracle, otherwise a benchmark comparison against it
/// would compare different tokenizations. Pre-tokenization is done by Atlas
/// (piecewise) because fastembed forces the onig regex backend onto the HF
/// crate via feature unification; see common::build_hf_cl100k.
#[test]
fn hf_tokenizer_derived_from_ranks_matches_oracle() {
    let oracle = cl100k_base().expect("oracle loads");
    let hf = build_hf_cl100k(&load_ranks());
    for text in sample_texts() {
        let hf_ids = common::hf_encode_piecewise(&hf, text);
        assert_eq!(
            hf_ids,
            oracle.encode_ordinary(text),
            "hf mismatch for {text:?}"
        );
    }
}
