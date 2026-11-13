//! Backend abstraction over the durable-record primitives.
//!
//! [`sqlite_store::SqliteStore`] implements [`Backend`]. The workflow layer
//! (`learn`, `search`, `hybrid_search`) is generic over the trait, which
//! kept the PoC parity tests honest and keeps future backends possible
//! without touching pipeline code.
//!
//! The trait uses `-> impl Future + Send` signatures (static dispatch
//! only) — no `async-trait` dependency; Atlas never needs trait objects
//! here.
//!
//! Idempotence contract (must hold for every backend): every write is an
//! upsert keyed by a deterministic identity, so re-running `learn` on the
//! same text is a no-op and a partial write is repaired by re-running it.

use crate::records::{
    ChunkRecord, DocumentRecord, EmbeddingRecord, EntityRecord, EvidenceRecord, PredicateRecord,
    RelationshipRecord,
};
use atlas_graph::KnowledgeGraph;
use std::future::Future;

/// Everything one `learn` call persists, grouped so a backend may write it
/// atomically. The SQLite write order is document, chunks, embeddings, then
/// graph entities, predicates, relationships, and their evidence.
pub struct LearnWrite<'a> {
    pub document: &'a DocumentRecord,
    pub chunks: &'a [ChunkRecord],
    pub embeddings: &'a [EmbeddingRecord],
    pub graph: &'a KnowledgeGraph,
}

pub trait Backend: Send + Sync {
    fn upsert_document(
        &self,
        doc_id: &str,
        text: &str,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
    fn document_exists(&self, doc_id: &str) -> impl Future<Output = anyhow::Result<bool>> + Send;
    fn upsert_chunk(&self, chunk: &ChunkRecord) -> impl Future<Output = anyhow::Result<()>> + Send;
    fn get_chunk(
        &self,
        chunk_id: &str,
    ) -> impl Future<Output = anyhow::Result<Option<ChunkRecord>>> + Send;
    fn delete_chunk(&self, chunk_id: &str) -> impl Future<Output = anyhow::Result<()>> + Send;
    fn upsert_embedding(
        &self,
        embedding: &EmbeddingRecord,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
    /// All embeddings for one model, used to rebuild the search index.
    fn load_embeddings(
        &self,
        model_id: &str,
    ) -> impl Future<Output = anyhow::Result<Vec<EmbeddingRecord>>> + Send;
    fn upsert_entity(
        &self,
        entity: &EntityRecord,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
    fn upsert_predicate(
        &self,
        predicate: &PredicateRecord,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
    /// Upsert a relationship *including* its full evidence set: an
    /// idempotent replace of the relationship row and its evidence.
    fn upsert_relationship(
        &self,
        rel: &RelationshipRecord,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
    fn load_graph(&self) -> impl Future<Output = anyhow::Result<KnowledgeGraph>> + Send;

    /// Check the expected fixed-layout records for this document and model.
    /// Backends without a keyed check conservatively replay the idempotent group.
    fn learn_is_complete(
        &self,
        _doc_id: &str,
        _model_id: &str,
        _chunks: &[ChunkRecord],
        _graph: &KnowledgeGraph,
    ) -> impl Future<Output = anyhow::Result<bool>> + Send {
        async { Ok(false) }
    }

    /// Persist everything one `learn` call produced. The graph is a
    /// document-local delta: merge its evidence without removing earlier
    /// sources, preserve relationship order, and never downgrade origin.
    ///
    /// Implementations must serialize evidence merges with other writers.
    /// The replacement primitives cannot provide a safe default merge.
    /// [`SqliteStore`] commits the whole group in one transaction.
    fn apply_learn_write(
        &self,
        write: &LearnWrite<'_>,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    /// Persist a whole graph: entities, predicates, relationships — all as
    /// idempotent upserts keyed by normalized identity. Identical loop for
    /// every backend because it is expressed through the primitives above.
    fn persist_graph(
        &self,
        graph: &KnowledgeGraph,
    ) -> impl Future<Output = anyhow::Result<()>> + Send {
        async move {
            for (ordinal, (_, entity)) in graph.entities().enumerate() {
                self.upsert_entity(&EntityRecord {
                    // The normalized key is the identity; the display form
                    // normalizes back to it.
                    key: atlas_graph::normalize(&entity.display),
                    display: entity.display.clone(),
                    kind: entity.kind.clone(),
                    ordinal: ordinal as u32,
                })
                .await?;
            }
            for (ordinal, (_, key)) in graph.predicates().enumerate() {
                self.upsert_predicate(&PredicateRecord {
                    key: key.to_string(),
                    ordinal: ordinal as u32,
                })
                .await?;
            }
            for (ordinal, (_, rel)) in graph.relationships().enumerate() {
                self.upsert_relationship(&RelationshipRecord {
                    src: graph.entity_key(rel.src).ok_or_else(|| {
                        anyhow::anyhow!("relationship references unknown source entity")
                    })?,
                    predicate: graph
                        .predicate(rel.predicate)
                        .unwrap_or_default()
                        .to_string(),
                    dst: graph.entity_key(rel.dst).ok_or_else(|| {
                        anyhow::anyhow!("relationship references unknown destination entity")
                    })?,
                    origin: crate::records::origin_to_str(rel.origin).to_string(),
                    evidence: rel
                        .evidence
                        .iter()
                        .map(|e: &atlas_graph::Evidence| EvidenceRecord {
                            doc_id: e.doc_id.clone(),
                            sentence: e.sentence.clone(),
                        })
                        .collect(),
                    ordinal: ordinal as u32,
                })
                .await?;
            }
            Ok(())
        }
    }
}
