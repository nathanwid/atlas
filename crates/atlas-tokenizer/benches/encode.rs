//! Encode-throughput benchmark of the two Atlas merge engines versus the
//! tiktoken-rs oracle and the HuggingFace `tokenizers` crate (M02).
//!
//! All four encode the identical input (the committed corpus fixture, ~10 KB
//! of plain English) end to end, including pre-tokenization, so the numbers
//! are directly comparable. Both Atlas engines are asserted equal to the
//! oracle by the integration tests; this bench measures cost, not
//! correctness.

#[path = "../tests/common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_encode(c: &mut Criterion) {
    let corpus = common::load_corpus();
    let ranks = common::load_ranks();
    let atlas = atlas_tokenizer::BpeTokenizer::from_ranks(ranks.clone());
    let oracle = tiktoken_rs::cl100k_base().expect("oracle loads");
    let hf = common::build_hf_cl100k(&ranks);

    let mut group = c.benchmark_group("cl100k_encode");
    group.throughput(criterion::Throughput::Bytes(corpus.len() as u64));

    group.bench_function("atlas_naive", |b| b.iter(|| atlas.encode_naive(&corpus)));
    group.bench_function("atlas_heap", |b| b.iter(|| atlas.encode(&corpus)));
    group.bench_function("tiktoken_rs", |b| {
        b.iter(|| oracle.encode_ordinary(&corpus))
    });
    group.bench_function("hf_tokenizers", |b| {
        b.iter(|| common::hf_encode_piecewise(&hf, &corpus))
    });

    group.finish();

    // The corpus is made of short pre-tokens (average word length), which is
    // the regime where a rescan beats heap bookkeeping. This second group
    // exercises the opposite regime: two very long pre-tokens (one letter
    // run, one punctuation run) where the number of merges per piece is
    // large enough for the O(log n) engine to matter.
    let long_pieces: String = "a".repeat(4096) + &"!".repeat(4096) + &"\u{1F600}".repeat(256);
    let mut group = c.benchmark_group("long_piece_encode");
    group.throughput(criterion::Throughput::Bytes(long_pieces.len() as u64));

    group.bench_function("atlas_naive", |b| {
        b.iter(|| atlas.encode_naive(&long_pieces))
    });
    group.bench_function("atlas_heap", |b| b.iter(|| atlas.encode(&long_pieces)));
    group.bench_function("tiktoken_rs", |b| {
        b.iter(|| oracle.encode_ordinary(&long_pieces))
    });
    group.bench_function("hf_tokenizers", |b| {
        b.iter(|| hf.encode(long_pieces.as_str(), false).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
