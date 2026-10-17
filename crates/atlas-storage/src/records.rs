//! Durable records stored in SurrealDB.
//!
//! Record identity is deterministic and content-derived where possible, so
//! re-learning the same text is a no-op:
//!
//! * `doc_id` = SHA-256 of the document text (hex).
//! * `chunk_id` = `{doc_id}:{ordinal}` (M02 chunking is deterministic).
//! * embedding key = `(chunk_id, model_id)` — different models never mix.
//! * relationship id = SHA-256 of the triple keys, joined with unit
//!   separators.
//!
//! Tables are schemaless; the record structs below are the schema.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub doc_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub chunk_id: String,
    pub doc_id: String,
    pub ordinal: u32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub chunk_id: String,
    pub model_id: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    /// Normalized identity key.
    pub key: String,
    /// First-seen display label.
    pub display: String,
    pub kind: Option<String>,
    /// Insertion order, so reconstruction reproduces first-seen IDs.
    pub ordinal: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateRecord {
    pub key: String,
    pub ordinal: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub doc_id: String,
    pub sentence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipRecord {
    /// Normalized source entity key.
    pub src: String,
    /// Normalized predicate key.
    pub predicate: String,
    /// Normalized destination entity key.
    pub dst: String,
    /// "explicit" or "co_occurrence".
    pub origin: String,
    pub evidence: Vec<EvidenceRecord>,
    pub ordinal: u32,
}

/// Content-addressed document identity.
pub fn doc_id_for(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

/// Deterministic relationship identity from the triple's natural keys.
pub fn relationship_id(src: &str, predicate: &str, dst: &str) -> String {
    let mut hasher = Sha256::new();
    for part in [src, predicate, dst] {
        hasher.update(part.as_bytes());
        hasher.update([0x1F]); // unit separator keeps parts unambiguous
    }
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Map a graph [`EdgeOrigin`](atlas_graph::EdgeOrigin) to its stored form.
pub fn origin_to_str(origin: atlas_graph::EdgeOrigin) -> &'static str {
    match origin {
        atlas_graph::EdgeOrigin::Explicit => "explicit",
        atlas_graph::EdgeOrigin::CoOccurrence => "co_occurrence",
    }
}

/// Parse a stored origin string.
pub fn origin_from_str(s: &str) -> Option<atlas_graph::EdgeOrigin> {
    match s {
        "explicit" => Some(atlas_graph::EdgeOrigin::Explicit),
        "co_occurrence" => Some(atlas_graph::EdgeOrigin::CoOccurrence),
        _ => None,
    }
}
