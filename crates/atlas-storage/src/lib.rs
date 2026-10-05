//! Vector indexes for Atlas (M04 similarity-search layer).
//!
//! [`index::VectorIndex`] is the boundary; [`index::BruteForceIndex`] is its
//! first implementation: an exact exhaustive scan over contiguous storage.
//! It is deliberately the simplest possible index: slow in the large, exact
//! everywhere, and the baseline of truth that any future index work must be
//! measured against. Durable storage backends belong to M07.

pub mod brute;
pub mod index;

pub use brute::BruteForceIndex;
pub use index::{SearchHit, VectorIndex};
