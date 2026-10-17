//! Durable storage for Atlas (M07).
//!
//! [`store::Store`] wraps an embedded SurrealDB (in-memory for tests,
//! RocksDB at `<data_dir>/surreal` for durability). [`crate::records`]
//! define the deterministic record identities; [`workflow::learn`] and
//! [`workflow::search`] are the durable flows:
//!
//! ```text
//! learn:  input → document → chunks → embeddings → graph → persistence
//! search: query → embed_query → persisted embeddings → BruteForceIndex
//! ```
//!
//! All writes are idempotent upserts keyed by content-derived IDs, so
//! duplicate `learn` runs are no-ops and partial failures are repaired by
//! re-running. The vector-search decision: embeddings live in a table and
//! are loaded into the existing exact [`index::BruteForceIndex`] on demand
//! — no ANN machinery, appropriate at Atlas's scale, and the
//! [`index::VectorIndex`] abstraction stays the only search surface.
//! Persistence of the M05 learned-model file is unchanged (file-based);
//! the `embedding` table is model-keyed, so learned vectors can join it
//! later without schema changes.

pub mod brute;
pub mod index;
pub mod records;
pub mod store;
pub mod workflow;

pub use brute::BruteForceIndex;
pub use index::{SearchHit, VectorIndex};
pub use records::{
    ChunkRecord, DocumentRecord, EmbeddingRecord, EntityRecord, EvidenceRecord, PredicateRecord,
    RelationshipRecord, doc_id_for, origin_from_str, origin_to_str,
};
pub use store::Store;
pub use workflow::{LearnSummary, SearchResult, learn, search};
