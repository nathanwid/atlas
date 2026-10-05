//! Vector indexes for Atlas.
//!
//! Milestone 3 introduces the [`index::VectorIndex`] boundary and its first
//! implementation: [`index::BruteForceIndex`], an exact exhaustive scan over
//! contiguous storage. It is deliberately the simplest possible index: slow
//! in the large, exact everywhere, and the baseline of truth that every
//! later index (SIMD scan, HNSW, pgvector) must be measured against.

pub mod brute;
pub mod index;

pub use brute::BruteForceIndex;
pub use index::{SearchHit, VectorIndex};
