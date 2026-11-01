//! Durable storage for Atlas.
//!
//! [`sqlite_store::SqliteStore`] is the production backend: an embedded
//! SQLite database (bundled via rusqlite, WAL journal, `synchronous=FULL`)
//! at `<data_dir>/atlas.db`. [`crate::records`] define the deterministic
//! record identities; [`workflow::learn`] and [`workflow::search`] are the
//! durable flows:
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

pub mod backend;
pub mod brute;
pub mod hybrid;
pub mod index;
pub mod records;
pub mod sqlite_store;
pub mod workflow;

pub use backend::Backend;
pub use backend::LearnWrite;
pub use brute::BruteForceIndex;
pub use hybrid::{HybridConfig, HybridResult, ResultSource, SearchMode, hybrid_search};
pub use index::{SearchHit, VectorIndex};
pub use records::{
    ChunkRecord, DocumentRecord, EmbeddingRecord, EntityRecord, EvidenceRecord, PredicateRecord,
    RelationshipRecord, doc_id_for, origin_from_str, origin_to_str,
};
pub use sqlite_store::SqliteStore;
pub use workflow::{LearnSummary, SearchResult, learn, search};
