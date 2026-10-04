//! Disk cache for chunk embeddings.
//!
//! Embedding a chunk costs a model forward pass; embedding the same chunk
//! again should cost a file read. The cache is content-addressed: the key is
//! SHA-256(model id + chunk text), so a changed model or a changed chunk can
//! never collide with stale vectors.
//!
//! The whole cache is one bincode file using a `BTreeMap`, which makes the
//! serialized bytes deterministic (no map-iteration order) and therefore
//! diffable. A missing or corrupted file is treated as an empty cache:
//! the cache is disposable by definition and must never take the pipeline
//! down.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// SHA-256 digest length.
pub type CacheKey = [u8; 32];

/// Content-addressed cache key: model id and chunk text are separated by a
/// zero byte so the pair is unambiguous.
pub fn cache_key(model_id: &str, text: &str) -> CacheKey {
    let mut hasher = Sha256::new();
    hasher.update(model_id.as_bytes());
    hasher.update([0u8]);
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

#[derive(Debug, Default)]
pub struct VectorCache {
    path: PathBuf,
    entries: BTreeMap<Vec<u8>, Vec<f32>>,
}

impl VectorCache {
    /// Load from disk. A missing or corrupt file yields an empty cache.
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let entries = std::fs::read(&path)
            .ok()
            .and_then(|bytes| bincode::deserialize::<BTreeMap<Vec<u8>, Vec<f32>>>(&bytes).ok())
            .unwrap_or_default();
        VectorCache { path, entries }
    }

    pub fn get(&self, key: &CacheKey) -> Option<&Vec<f32>> {
        self.entries.get(key.as_slice())
    }

    /// Insert a vector. Returns true if the key was not present before.
    pub fn insert(&mut self, key: CacheKey, vector: Vec<f32>) -> bool {
        self.entries.insert(key.to_vec(), vector).is_none()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Write the cache to disk. Direct write, no temp-file dance: the cache
    /// is rebuilt from scratch if a torn write corrupts it.
    pub fn save(&self) -> std::io::Result<()> {
        let bytes = bincode::serialize(&self.entries)
            .expect("BTreeMap<Vec<u8>, Vec<f32>> is always serializable");
        std::fs::write(&self.path, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_depends_on_model_and_text() {
        let a = cache_key("m", "hello");
        let b = cache_key("m", "hello");
        let c = cache_key("m2", "hello");
        let d = cache_key("m", "hell");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.bin");
        let key = cache_key("model", "text");

        let mut cache = VectorCache::load(&path);
        assert!(cache.is_empty());
        cache.insert(key, vec![1.0, -2.5]);
        cache.save().unwrap();

        let reloaded = VectorCache::load(&path);
        assert_eq!(reloaded.len(), 1);
        assert_eq!(reloaded.get(&key), Some(&vec![1.0, -2.5]));
    }

    #[test]
    fn missing_file_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cache = VectorCache::load(dir.path().join("does-not-exist.bin"));
        assert!(cache.is_empty());
    }

    #[test]
    fn corrupt_file_loads_empty_instead_of_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.bin");
        std::fs::write(&path, b"not a bincode file at all").unwrap();
        let cache = VectorCache::load(&path);
        assert!(cache.is_empty());
    }

    #[test]
    fn reinsert_same_key_replaces() {
        let mut cache = VectorCache::load("unused");
        let key = cache_key("m", "t");
        assert!(cache.insert(key, vec![1.0]));
        assert!(!cache.insert(key, vec![2.0]));
        assert_eq!(cache.get(&key), Some(&vec![2.0]));
    }
}
