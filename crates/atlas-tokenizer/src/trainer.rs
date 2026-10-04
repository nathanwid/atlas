//! BPE training over a corpus.
//!
//! Training is the mirror of encoding. Starting from the 256 single-byte
//! tokens, the loop is:
//!
//! 1. Pre-tokenize the corpus with the same pattern used at encode time (the
//!    split points must agree, or encoding would replay merges across
//!    boundaries training never saw).
//! 2. Count every adjacent symbol pair across distinct pieces, weighted by
//!    how often each piece occurs.
//! 3. Merge the most frequent pair into a new token with the next rank.
//! 4. Repeat until the vocabulary reaches the requested size.
//!
//! Two implementation notes that matter for correctness:
//!
//! * Ties in pair counts are broken deterministically (highest count, then
//!   smallest pair), so training the same corpus twice yields the exact same
//!   table — the determinism test asserts this.
//! * Two different merges can produce the same byte string (e.g. `ab` + `c`
//!   and `a` + `bc` both produce `abc`). A table cannot hold duplicates, so
//!   the second merge to produce existing bytes reuses the existing rank
//!   instead of creating one.

use crate::encoder::BpeTokenizer;
use crate::pretokenize::pretokenize;
use crate::ranks::{Rank, Ranks};
use std::collections::HashMap;

/// Train a BPE tokenizer on `text` until the vocabulary holds
/// `target_vocab_size` tokens (minimum 256: all single bytes).
///
/// Training stops early if no pair occurs at least twice; merging
/// once-occurring pairs would grow the vocabulary without helping any future
/// input compress.
pub fn train(text: &str, target_vocab_size: usize) -> BpeTokenizer {
    assert!(
        target_vocab_size >= 256,
        "a byte-level vocabulary always contains the 256 single-byte tokens"
    );

    // Distinct pieces with counts. Weighting by piece frequency is what
    // makes the vocabulary fit the corpus; without it, every rare spelling
    // would get merges it deserves no more than any other.
    let mut freqs: HashMap<&str, u64> = HashMap::new();
    for piece in pretokenize(text) {
        *freqs.entry(piece).or_insert(0) += 1;
    }

    // Bytes for every rank so far; single bytes are ranks 0..256.
    let mut vocab_bytes: Vec<Vec<u8>> = (0..=255u32).map(|b| vec![b as u8]).collect();
    // Each distinct piece as a sequence of symbol ranks, plus its count.
    let mut words: Vec<(Vec<Rank>, u64)> = freqs
        .into_iter()
        .map(|(piece, count)| {
            let symbols = piece.bytes().map(|b| b as Rank).collect();
            (symbols, count)
        })
        .collect();

    while vocab_bytes.len() < target_vocab_size {
        let mut pairs: HashMap<(Rank, Rank), u64> = HashMap::new();
        for (symbols, count) in &words {
            for pair in symbols.windows(2) {
                *pairs.entry((pair[0], pair[1])).or_insert(0) += count;
            }
        }
        // Most frequent pair; ties go to the smallest pair for determinism.
        let best = pairs
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .max_by_key(|((a, b), count)| (*count, std::cmp::Reverse((*a, *b))))
            .map(|(pair, _)| pair);
        let Some((left, right)) = best else { break };

        let mut merged = vocab_bytes[left as usize].clone();
        merged.extend_from_slice(&vocab_bytes[right as usize]);

        // Another merge may already have created these bytes. If so, reuse
        // that rank: the table cannot hold two ranks for one token.
        let existing = vocab_bytes.iter().position(|b| *b == merged);
        let new_rank = match existing {
            Some(r) => r as Rank,
            None => {
                let r = vocab_bytes.len() as Rank;
                vocab_bytes.push(merged);
                r
            }
        };

        // Apply the merge left-to-right inside every piece. A fresh rank can
        // never equal either of its parents, so a single pass cannot miss a
        // recursive overlap like "aaa".
        for (symbols, _) in &mut words {
            let mut i = 0;
            while i + 1 < symbols.len() {
                if (symbols[i], symbols[i + 1]) == (left, right) {
                    symbols.remove(i + 1);
                    symbols[i] = new_rank;
                }
                i += 1;
            }
        }
    }

    let ranks: Ranks = vocab_bytes
        .into_iter()
        .enumerate()
        .map(|(rank, bytes)| (bytes.into(), rank as Rank))
        .collect();
    BpeTokenizer::from_ranks(ranks)
}
