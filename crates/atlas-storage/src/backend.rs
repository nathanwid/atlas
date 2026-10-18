//! Backend abstraction over the durable-record primitives.
//!
//! [`store::Store`] (embedded SurrealDB) and, under the opt-in
//! `sqlite-backend` feature, [`sqlite_store::SqliteStore`] both implement
//! [`Backend`]. The workflow layer (`learn`, `search`, `hybrid_search`) is
//! generic over it, so the PoC parity tests exercise the exact same
//! pipeline code against both engines and a future default-backend flip is
//! a constructor change, not a logic change.
//!
//! The trait uses native `async fn` (static dispatch only) — no
//! `async-trait` dependency; Atlas never needs trait objects here.
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
/// atomically. Record order inside the group matches the historical write
/// order: document, then (chunk, embedding) per chunk, then the graph's
/// entities, predicates, and relationships with their evidence.
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

    /// Persist everything one `learn` call produced.
    ///
    /// The default writes each record through the primitives in the
    /// historical order — one autocommit per record, exactly the
    /// pre-batching behavior. Transactional backends override this to
    /// commit the whole group in a single transaction; the durability
    /// contract (all-or-nothing per group, or idempotently repairable
    /// partial state) holds either way because every record is an
    /// idempotent upsert keyed by a deterministic identity.
    fn apply_learn_write(
        &self,
        write: &LearnWrite<'_>,
    ) -> impl Future<Output = anyhow::Result<()>> + Send {
        async move {
            self.upsert_document(&write.document.doc_id, &write.document.text)
                .await?;
            for chunk in write.chunks {
                self.upsert_chunk(chunk).await?;
            }
            for embedding in write.embeddings {
                self.upsert_embedding(embedding).await?;
            }
            self.persist_graph(write.graph).await?;
            Ok(())
        }
    }

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
