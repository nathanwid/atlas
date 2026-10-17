//! Atlas's knowledge graph (M06): entities, directed relationships, and
//! deterministic extraction and traversal.
//!
//! Ownership boundary: this crate is self-contained — std only, no
//! dependencies on other Atlas crates. Entities are surface terms with
//! normalized lowercase identity and first-seen display labels; BPE tokens
//! and vocabulary IDs are a different representation and stay in their own
//! crates.
//!
//! Extraction (`extract`) is deliberately simple and deterministic: a
//! predicate lexicon produces `Explicit` relationships, same-sentence
//! co-occurrence produces lower-confidence `CoOccurrence` fallback edges,
//! and evidence (document + sentence) merges on duplicates. No LLM, no NLP
//! libraries.
//!
//! Traversal (`neighbors`, `reachable`) is insertion-order deterministic.
//! What is *not* here, on purpose: persistence (M07 — the graph is plain
//! structs), graph scoring and personalized walks (M08 hybrid retrieval),
//! and any performance machinery.

pub mod extract;
pub mod graph;
pub mod pagerank;

pub use extract::{Extractor, ExtractorConfig};
pub use graph::{
    Direction, EdgeOrigin, Entity, EntityId, Evidence, KnowledgeGraph, Neighbor, PredicateId,
    Relationship, RelationshipId, normalize,
};
pub use pagerank::personalized_pagerank;
