//! Document chunking and embedding for Atlas.
//!
//! Milestone 2 pipeline: a document goes through
//!
//! ```text
//! document -> Chunker (token-bounded, pre-token-aligned chunks)
//!          -> Embedder (batched text -> vectors, pluggable)
//!          -> VectorCache (content-addressed, embed only on miss)
//! ```
//!
//! [`embed_chunks`] ties the three together and is the function later
//! milestones (indexing, retrieval) will call.

pub mod cache;
pub mod chunker;
pub mod embedder;

pub use cache::{CacheKey, VectorCache, cache_key};
pub use chunker::{Chunk, Chunker};
pub use embedder::{Embedder, FastembedEmbedder};

/// Chunk-cache-embed a batch of chunk texts.
///
/// For each chunk: hit the cache, or embed it (in one batched call for all
/// misses) and store the vector. The cache is saved only when at least one
/// vector was newly computed, so repeated runs do no I/O writes and no model
/// calls. Output order matches input order.
pub fn embed_chunks(
    embedder: &dyn Embedder,
    cache: &mut VectorCache,
    texts: &[&str],
) -> anyhow::Result<Vec<Vec<f32>>> {
    let keys: Vec<CacheKey> = texts
        .iter()
        .map(|t| cache_key(embedder.model_id(), t))
        .collect();

    let mut out: Vec<Option<Vec<f32>>> = texts.iter().map(|_| None).collect();
    let mut misses = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        match cache.get(key) {
            Some(vector) => out[i] = Some(vector.clone()),
            None => misses.push(i),
        }
    }

    if !misses.is_empty() {
        let missing: Vec<&str> = misses.iter().map(|&i| texts[i]).collect();
        let vectors = embedder.embed(&missing)?;
        anyhow::ensure!(
            vectors.len() == misses.len(),
            "embedder returned {} vectors for {} texts",
            vectors.len(),
            misses.len()
        );
        for (i, vector) in misses.into_iter().zip(vectors) {
            cache.insert(keys[i], vector.clone());
            out[i] = Some(vector);
        }
        cache.save()?;
    }

    Ok(out.into_iter().map(Option::unwrap).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake embedder that never touches a model: the vector is derived from
    /// the text, and every call is counted so tests can assert exactly how
    /// many embeds happened.
    struct CountingEmbedder {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl CountingEmbedder {
        fn new() -> Self {
            CountingEmbedder {
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl Embedder for CountingEmbedder {
        fn model_id(&self) -> &str {
            "counting-fake"
        }

        fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, t.starts_with('a') as u8 as f32])
                .collect())
        }
    }

    #[test]
    fn second_run_performs_zero_embedder_calls() {
        let embedder = CountingEmbedder::new();
        let dir = tempfile::tempdir().unwrap();
        let mut cache = VectorCache::load(dir.path().join("cache.bin"));

        let texts = vec!["alpha", "beta", "gamma"];
        let first = embed_chunks(&embedder, &mut cache, &texts).unwrap();
        assert_eq!(embedder.call_count(), 1, "first run must embed the misses");
        assert_eq!(cache.len(), 3);

        // Same texts again: all hits, no embed call, cache unchanged.
        let second = embed_chunks(&embedder, &mut cache, &texts).unwrap();
        assert_eq!(
            embedder.call_count(),
            1,
            "second run must be pure cache hits"
        );
        assert_eq!(first, second);

        // One new text: only the miss is embedded, hits keep their vectors.
        let grown = embed_chunks(&embedder, &mut cache, &["alpha", "delta"]).unwrap();
        assert_eq!(embedder.call_count(), 2);
        assert_eq!(grown[0], first[0]);
        assert_eq!(
            grown[1],
            vec![5.0, 0.0],
            "delta: len 5, does not start with 'a'"
        );
    }

    #[test]
    fn output_order_matches_input_order_despite_cache_hits() {
        let embedder = CountingEmbedder::new();
        let dir = tempfile::tempdir().unwrap();
        let mut cache = VectorCache::load(dir.path().join("cache.bin"));

        let _ = embed_chunks(&embedder, &mut cache, &["only"]).unwrap();
        let mixed = embed_chunks(&embedder, &mut cache, &["new", "only"]).unwrap();
        assert_eq!(mixed[0], vec![3.0, 0.0]);
        assert_eq!(
            mixed[1],
            vec![4.0, 0.0],
            "cached vector for 'only' is reused as-is"
        );
    }

    #[test]
    fn empty_input_is_empty_output() {
        let embedder = CountingEmbedder::new();
        let dir = tempfile::tempdir().unwrap();
        let mut cache = VectorCache::load(dir.path().join("cache.bin"));
        let out = embed_chunks(&embedder, &mut cache, &[]).unwrap();
        assert!(out.is_empty());
        assert_eq!(embedder.call_count(), 0);
    }
}
