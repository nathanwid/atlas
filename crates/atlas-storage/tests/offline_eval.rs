//! Offline evaluation: harness mechanics with vectors whose ground truth is
//! known by construction. No model, no network, fully deterministic.
//!
//! The fake embedder marks, for query `j`, the bit `j` in any text that is
//! query `j` itself or contains its first expected substring. Therefore
//! exactly one bit is set per query vector, and a chunk's score against its
//! own query depends only on how many marked bits its chunk carries:
//! chunks containing a single expected substring score 1.0; chunks holding
//! two queries' substrings score 1/sqrt(2) each. This makes the expected
//! recall computable by hand.

mod common;

use atlas_embedding::{Embedder, VectorCache, embed_chunks};
use atlas_storage::{BruteForceIndex, VectorIndex};
use common::{EvalQuery, load_eval_set, recall_at, resolve_expected};

/// Offline stand-in for `FastembedEmbedder`: no model, deterministic.
struct MarkingEmbedder {
    queries: Vec<EvalQuery>,
}

impl Embedder for MarkingEmbedder {
    fn model_id(&self) -> &str {
        "marking-fake"
    }

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let dim = self.queries.len();
        Ok(texts
            .iter()
            .map(|text| {
                let text = common::squash_ws(text);
                let mut v = vec![0.0; dim];
                for (j, q) in self.queries.iter().enumerate() {
                    let is_query = text == common::squash_ws(&q.query);
                    let holds_marker = q
                        .expected
                        .iter()
                        .any(|e| text.contains(&common::squash_ws(e)));
                    if is_query || holds_marker {
                        v[j] = 1.0;
                    }
                }
                v
            })
            .collect())
    }
}

#[test]
fn recall_math_is_exact_on_constructed_vectors() {
    // 4 one-hot vectors, dim 4: vector i == e_i.
    let vectors: Vec<Vec<f32>> = (0..4)
        .map(|i| (0..4).map(|j| if j == i { 1.0 } else { 0.0 }).collect())
        .collect();
    let index = BruteForceIndex::build(4, &vectors).unwrap();

    let query_e0 = vec![1.0, 0.0, 0.0, 0.0];
    let hits = index.search(&query_e0, 4);
    // Closest to e0 is vector 0 (score 1), then the orthogonal ones, ids
    // ascending among ties: 1, 2, 3, then the opposite vector 3 is not
    // orthogonal... e3 has a 0.0 dot with e0, all of 1..3 score 0.
    assert_eq!(hits[0].id, 0);

    assert!((recall_at(&hits, &[0], 1) - 1.0).abs() < 1e-9);
    assert!((recall_at(&hits, &[0, 3], 1) - 0.5).abs() < 1e-9);
    assert!((recall_at(&hits, &[0, 3], 4) - 1.0).abs() < 1e-9);
    assert!((recall_at(&hits, &[1], 1) - 0.0).abs() < 1e-9);
}

#[test]
fn eval_harness_runs_over_corpus_chunks_offline() {
    let eval = load_eval_set();
    let (_tokenizer, chunks) = common::load_corpus_chunks(128);
    let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();

    let embedder = MarkingEmbedder {
        queries: eval.queries.clone(),
    };
    let dir = tempfile::tempdir().unwrap();
    let mut cache = VectorCache::load(dir.path().join("offline.bin"));
    let vectors = embed_chunks(&embedder, &mut cache, &chunk_refs).unwrap();
    let dim = vectors[0].len();
    let index = BruteForceIndex::build(dim, &vectors).unwrap();

    let mut expected_all = Vec::new();
    for q in &eval.queries {
        expected_all.push(resolve_expected(&chunk_refs, q));
    }

    for (j, _q) in eval.queries.iter().enumerate() {
        let hits = index.search(&vectors_that_encode_query(&embedder, j), 5);
        assert_eq!(hits.len(), 5.min(chunks.len()));
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
        for hit in &hits {
            assert!(hit.score.is_finite());
            assert!((hit.id as usize) < chunks.len());
        }

        let expected = &expected_all[j];
        let r1 = recall_at(&hits, expected, 1);
        let r5 = recall_at(&hits, expected, 5);
        println!("offline q{j}: recall@1={r1:.3} recall@5={r5:.3} expected={expected:?}");

        // Ground truth by construction: exactly one chunk carries this
        // query's bit, and the query vector is that same unit bit, so the
        // marked chunk must be the unique top hit.
        assert!(
            (r1 - 1.0).abs() < 1e-9,
            "q{j}: marked chunk must rank first by construction (recall@1={r1})"
        );
        assert!((r5 - 1.0).abs() < 1e-9);
    }
}

/// The query vector for query j: one-hot at bit j, same encoding the fake
/// embedder gives to the query text itself.
fn vectors_that_encode_query(embedder: &MarkingEmbedder, j: usize) -> Vec<f32> {
    embedder
        .embed_documents(&[&embedder.queries[j].query])
        .unwrap()
        .remove(0)
}
