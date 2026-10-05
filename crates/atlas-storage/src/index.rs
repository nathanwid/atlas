//! The `VectorIndex` boundary.
//!
//! Anything that can answer "which stored vectors are closest to this
//! query" implements this trait. Implementations choose their own accuracy
//! and speed trade-offs; the evaluation harness (see `tests/`) is what
//! makes those trade-offs comparable.

/// One search result: the id the vector was registered under and its
/// similarity to the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchHit {
    pub id: u32,
    pub score: f32,
}

pub trait VectorIndex {
    /// Return up to `k` hits, best first.
    ///
    /// The query must have the same dimensionality as the stored vectors;
    /// implementations may treat a mismatch as a contract violation.
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchHit>;

    /// Number of vectors stored.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimensionality of the stored vectors.
    fn dim(&self) -> usize;
}
