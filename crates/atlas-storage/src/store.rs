//! The durable store: an embedded SurrealDB handle plus table access.
//!
//! Engine choices (see workspace Cargo.toml features): `mem` for fast
//! tests, embedded SurrealKV at `<data_dir>/surreal` for durability
//! (pure-Rust engine; RocksDB was rejected because its C++ build requires
//! libclang). Tables are schemaless; [`crate::records`] defines the record
//! shapes and their deterministic identities. No extra indexes are defined:
//! reads are either by record id or full scans, which is appropriate at
//! Atlas's current scale.

use crate::records::{
    ChunkRecord, DocumentRecord, EmbeddingRecord, EntityRecord, EvidenceRecord, PredicateRecord,
    RelationshipRecord,
};
use atlas_graph::{Evidence, KnowledgeGraph};
use std::path::Path;
use surrealdb::Surreal;
use surrealdb::engine::local::{Db, Mem, SurrealKV};

pub struct Store {
    db: Surreal<Db>,
}

impl Store {
    /// In-memory store for tests.
    pub async fn open_mem() -> anyhow::Result<Self> {
        let db = Surreal::new::<Mem>(()).await?;
        Store::init(db).await
    }

    /// Durable store backed by embedded SurrealKV at `path` (a directory).
    /// SurrealKV is SurrealDB's pure-Rust KV engine, so durability has no
    /// C++ toolchain requirements.
    pub async fn open_durable(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let db = Surreal::new::<SurrealKV>(path.as_ref().display().to_string()).await?;
        Store::init(db).await
    }

    async fn init(db: Surreal<Db>) -> anyhow::Result<Self> {
        db.use_ns("atlas").await?;
        db.use_db("atlas").await?;
        Ok(Store { db })
    }

    pub async fn upsert_document(&self, doc_id: &str, text: &str) -> anyhow::Result<()> {
        let _: Option<DocumentRecord> = self
            .db
            .update(("document", doc_id))
            .content(DocumentRecord {
                doc_id: doc_id.to_string(),
                text: text.to_string(),
            })
            .await?;
        Ok(())
    }

    pub async fn document_exists(&self, doc_id: &str) -> anyhow::Result<bool> {
        let found: Option<DocumentRecord> = self.db.select(("document", doc_id)).await?;
        Ok(found.is_some())
    }

    pub async fn upsert_chunk(&self, chunk: &ChunkRecord) -> anyhow::Result<()> {
        let _: Option<ChunkRecord> = self
            .db
            .update(("chunk", &chunk.chunk_id))
            .content(chunk.clone())
            .await?;
        Ok(())
    }

    pub async fn get_chunk(&self, chunk_id: &str) -> anyhow::Result<Option<ChunkRecord>> {
        Ok(self.db.select(("chunk", chunk_id)).await?)
    }

    pub async fn upsert_embedding(&self, embedding: &EmbeddingRecord) -> anyhow::Result<()> {
        // Embedding key is (chunk_id, model_id); record id joins both.
        let id = format!("{}:{}", embedding.model_id, embedding.chunk_id);
        let _: Option<EmbeddingRecord> = self
            .db
            .update(("embedding", &id))
            .content(embedding.clone())
            .await?;
        Ok(())
    }

    /// All embeddings for one model, used to rebuild the search index.
    pub async fn load_embeddings(&self, model_id: &str) -> anyhow::Result<Vec<EmbeddingRecord>> {
        let all: Vec<EmbeddingRecord> = self.db.select("embedding").await?;
        Ok(all.into_iter().filter(|e| e.model_id == model_id).collect())
    }

    pub async fn upsert_entity(&self, entity: &EntityRecord) -> anyhow::Result<()> {
        let _: Option<EntityRecord> = self
            .db
            .update(("entity", &entity.key))
            .content(entity.clone())
            .await?;
        Ok(())
    }

    pub async fn upsert_predicate(&self, predicate: &PredicateRecord) -> anyhow::Result<()> {
        let _: Option<PredicateRecord> = self
            .db
            .update(("predicate", &predicate.key))
            .content(predicate.clone())
            .await?;
        Ok(())
    }

    pub async fn upsert_relationship(&self, rel: &RelationshipRecord) -> anyhow::Result<()> {
        let id = crate::records::relationship_id(&rel.src, &rel.predicate, &rel.dst);
        let _: Option<RelationshipRecord> = self
            .db
            .update(("relationship", &id))
            .content(rel.clone())
            .await?;
        Ok(())
    }

    /// Persist a whole graph: entities, predicates, relationships — all as
    /// idempotent upserts keyed by normalized identity.
    pub async fn persist_graph(&self, graph: &KnowledgeGraph) -> anyhow::Result<()> {
        for (ordinal, (id, entity)) in graph.entities().enumerate() {
            let _ = id;
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
        for (ordinal, (id, key)) in graph.predicates().enumerate() {
            let _ = id;
            self.upsert_predicate(&PredicateRecord {
                key: key.to_string(),
                ordinal: ordinal as u32,
            })
            .await?;
        }
        for (ordinal, (id, rel)) in graph.relationships().enumerate() {
            let _ = id;
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
                    .map(|e: &Evidence| EvidenceRecord {
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

    /// Rebuild a graph from persisted rows, in insertion order. The public
    /// graph builders enforce M06's dedup/upgrade/evidence rules during
    /// reconstruction, so the rebuilt graph matches the persisted one.
    pub async fn load_graph(&self) -> anyhow::Result<KnowledgeGraph> {
        let mut entities: Vec<EntityRecord> = self.db.select("entity").await?;
        let mut predicates: Vec<PredicateRecord> = self.db.select("predicate").await?;
        let mut relationships: Vec<RelationshipRecord> = self.db.select("relationship").await?;
        entities.sort_by_key(|e| e.ordinal);
        predicates.sort_by_key(|p| p.ordinal);
        relationships.sort_by_key(|r| r.ordinal);

        let mut graph = KnowledgeGraph::new();
        for entity in &entities {
            graph.intern_entity(&entity.display);
        }
        for predicate in &predicates {
            graph.intern_predicate(&predicate.key);
        }
        for rel in &relationships {
            let origin = crate::records::origin_from_str(&rel.origin)
                .ok_or_else(|| anyhow::anyhow!("stored origin {:?} is invalid", rel.origin))?;
            let src_id = graph.find_entity(&rel.src).ok_or_else(|| {
                anyhow::anyhow!(
                    "stored relationship references unknown entity {rel:?}",
                    rel = rel.src
                )
            })?;
            let dst_id = graph.find_entity(&rel.dst).ok_or_else(|| {
                anyhow::anyhow!("stored relationship references unknown entity {}", rel.dst)
            })?;
            let pid = graph.find_predicate(&rel.predicate).ok_or_else(|| {
                anyhow::anyhow!(
                    "stored relationship references unknown predicate {}",
                    rel.predicate
                )
            })?;
            for ev in &rel.evidence {
                graph.link(
                    src_id,
                    pid,
                    dst_id,
                    origin,
                    Evidence {
                        doc_id: ev.doc_id.clone(),
                        sentence: ev.sentence.clone(),
                    },
                );
            }
        }
        Ok(graph)
    }
}
