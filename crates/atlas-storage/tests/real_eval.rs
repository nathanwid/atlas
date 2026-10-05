//! Real-model evaluation for the M04 similarity-search layer.
//!
//! Ignored by default (model weights + network). Run explicitly:
//! `cargo test -p atlas-storage --release -- --ignored --nocapture`
//!
//! REPORT-ONLY on quality: recall@1/3/5 and per-query scan latency are
//! printed for the milestone summary; no recall threshold is asserted. The
//! assertions are mechanical only (dimensions, counts, finite scores).

mod common;

use atlas_embedding::{Embedder, FastembedEmbedder, VectorCache, embed_chunks};
use atlas_storage::{BruteForceIndex, VectorIndex};
use common::{load_eval_set, recall_at, resolve_expected};
use std::time::Instant;

#[test]
#[ignore = "downloads ~100 MB model weights on first run; requires network"]
fn real_model_eval_report_only() {
    let eval = load_eval_set();
    let (_tokenizer, chunks) = common::load_corpus_chunks(128);
    let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
    println!("{} chunks (128 tokens each)", chunks.len());

    let embedder = FastembedEmbedder::bge_small_en_v1_5().expect("model loads");
    let dir = tempfile::tempdir().unwrap();
    let mut cache = VectorCache::load(dir.path().join("real_eval.bin"));
    let vectors = embed_chunks(&embedder, &mut cache, &chunk_refs).expect("embeds");
    let dim = vectors[0].len();
    println!("embeddings: dim={dim}");

    let index = BruteForceIndex::build(dim, &vectors).expect("index builds");

    let mut recalls = Vec::new();
    let mut latencies = Vec::new();
    for q in &eval.queries {
        let expected = resolve_expected(&chunk_refs, q);
        // Queries go through embed_query (BGE prefixes them); documents
        // were embedded raw via embed_chunks.
        let query_vec = &embedder.embed_query(&q.query).expect("query embeds");

        let start = Instant::now();
        let hits = index.search(query_vec, 5);
        let elapsed = start.elapsed();

        // Mechanical assertions only.
        assert_eq!(hits.len(), 5.min(chunks.len()));
        for hit in &hits {
            assert!((hit.id as usize) < chunks.len());
            assert!(hit.score.is_finite());
            assert!(
                hit.score.abs() <= 1.0 + 1e-4,
                "cosine out of range: {}",
                hit.score
            );
        }

        let r1 = recall_at(&hits, &expected, 1);
        let r3 = recall_at(&hits, &expected, 3);
        let r5 = recall_at(&hits, &expected, 5);
        recalls.push((r1, r3, r5));
        latencies.push(elapsed);
        println!(
            "q{:<2} recall@1={:.3} @3={:.3} @5={:.3}  scan={:>7.1} µs  {:?}",
            eval.queries
                .iter()
                .position(|x| x.query == q.query)
                .unwrap(),
            r1,
            r3,
            r5,
            elapsed.as_nanos() as f64 / 1000.0,
            q.query
        );
    }

    let n = recalls.len() as f64;
    let mean = |f: fn(&(f64, f64, f64)) -> f64| recalls.iter().map(f).sum::<f64>() / n;
    println!("---");
    println!(
        "mean recall@1={:.3} @3={:.3} @5={:.3} over {} queries",
        mean(|r| r.0),
        mean(|r| r.1),
        mean(|r| r.2),
        recalls.len()
    );
    let mean_us = latencies
        .iter()
        .map(|d| d.as_nanos() as f64 / 1000.0)
        .sum::<f64>()
        / n;
    let max_us = latencies
        .iter()
        .map(|d| d.as_nanos() as f64 / 1000.0)
        .fold(0.0, f64::max);
    println!(
        "scan latency: mean={mean_us:.1} µs max={max_us:.1} µs (brute force, {} x {} floats)",
        chunks.len(),
        dim
    );
}
