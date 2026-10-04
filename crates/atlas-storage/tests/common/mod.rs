//! Shared evaluation helpers for the M3 tests.
//!
//! Included by both integration tests via `mod common;`. Each test binary
//! uses a different subset, hence the module-wide dead_code allowance.

#![allow(dead_code)]

use atlas_storage::SearchHit;
use serde::Deserialize;

#[derive(Clone, Deserialize)]
pub struct EvalQuery {
    pub query: String,
    /// Substrings that identify known-good chunks. A chunk is expected if
    /// its text contains the substring.
    pub expected: Vec<String>,
}

#[derive(Deserialize)]
pub struct EvalSet {
    pub queries: Vec<EvalQuery>,
}

pub fn load_eval_set() -> EvalSet {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/eval_set.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("eval set fixture"))
        .expect("eval set fixture parses")
}

pub fn load_corpus() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/corpus.txt"
    );
    std::fs::read_to_string(path).expect("corpus fixture")
}

pub fn load_corpus_chunks(max_tokens: usize) -> (atlas_tokenizer::BpeTokenizer, Vec<String>) {
    let corpus = load_corpus();
    let ranks = std::fs::File::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
    ))
    .expect("ranks fixture");
    let tokenizer =
        atlas_tokenizer::BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(ranks))
            .expect("ranks parse");
    let chunks = atlas_embedding::Chunker::new(&tokenizer, max_tokens, 0)
        .chunk(&corpus)
        .into_iter()
        .map(|c| c.text.to_string())
        .collect();
    (tokenizer, chunks)
}

/// Collapse all whitespace runs to single spaces so expected substrings are
/// matched against content, not line wrapping.
pub fn squash_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Map expected substrings to chunk ids. Matching is whitespace-normalized.
/// Panics listing every unresolvable substring: fixture drift (corpus or
/// chunking changed) must be loud, a silent zero would quietly deflate
/// recall.
pub fn resolve_expected(chunks: &[&str], query: &EvalQuery) -> Vec<u32> {
    let squashed: Vec<String> = chunks.iter().map(|c| squash_ws(c)).collect();
    let mut ids = Vec::new();
    let mut missing = Vec::new();
    for expected in &query.expected {
        let needle = squash_ws(expected);
        match squashed.iter().position(|chunk| chunk.contains(&needle)) {
            Some(id) => {
                if !ids.contains(&(id as u32)) {
                    ids.push(id as u32);
                }
            }
            None => missing.push(expected.clone()),
        }
    }
    if !missing.is_empty() {
        panic!(
            "expected substrings not found in any single chunk: {missing:?}\n\
             the substring may straddle a chunk boundary or the corpus changed;\n\
             shorten the substring or update eval_set.json"
        );
    }
    ids
}

/// recall@k = |top-k ∩ expected| / |expected|.
pub fn recall_at(hits: &[SearchHit], expected: &[u32], k: usize) -> f64 {
    if expected.is_empty() {
        return 0.0;
    }
    let top: std::collections::HashSet<u32> = hits.iter().take(k).map(|h| h.id).collect();
    let found = expected.iter().filter(|id| top.contains(id)).count();
    found as f64 / expected.len() as f64
}
