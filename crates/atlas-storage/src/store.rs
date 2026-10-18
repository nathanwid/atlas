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
    ChunkRecord, DocumentRecord, EmbeddingRecord, EntityRecord, PredicateRecord, RelationshipRecord,
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

    /// Remove one chunk row. Used by hardening tests to simulate partial
    /// states; evidence referencing the deleted chunk is skipped by
    /// retrieval and QA.
    pub async fn delete_chunk(&self, chunk_id: &str) -> anyhow::Result<()> {
        let _: Option<ChunkRecord> = self.db.delete(("chunk", chunk_id)).await?;
        Ok(())
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

/// The SurrealDB store as a workflow [`Backend`]: every trait method
/// delegates to the identical inherent method (inherent methods win in
/// method resolution, so the bodies below call the ones defined above).
/// `persist_graph` is inherited from the trait default, expressed through
/// the upsert primitives.
impl crate::backend::Backend for Store {
    async fn upsert_document(&self, doc_id: &str, text: &str) -> anyhow::Result<()> {
        Store::upsert_document(self, doc_id, text).await
    }

    async fn document_exists(&self, doc_id: &str) -> anyhow::Result<bool> {
        Store::document_exists(self, doc_id).await
    }

    async fn upsert_chunk(&self, chunk: &ChunkRecord) -> anyhow::Result<()> {
        Store::upsert_chunk(self, chunk).await
    }

    async fn get_chunk(&self, chunk_id: &str) -> anyhow::Result<Option<ChunkRecord>> {
        Store::get_chunk(self, chunk_id).await
    }

    async fn delete_chunk(&self, chunk_id: &str) -> anyhow::Result<()> {
        Store::delete_chunk(self, chunk_id).await
    }

    async fn upsert_embedding(&self, embedding: &EmbeddingRecord) -> anyhow::Result<()> {
        Store::upsert_embedding(self, embedding).await
    }

    async fn load_embeddings(&self, model_id: &str) -> anyhow::Result<Vec<EmbeddingRecord>> {
        Store::load_embeddings(self, model_id).await
    }

    async fn upsert_entity(&self, entity: &EntityRecord) -> anyhow::Result<()> {
        Store::upsert_entity(self, entity).await
    }

    async fn upsert_predicate(&self, predicate: &PredicateRecord) -> anyhow::Result<()> {
        Store::upsert_predicate(self, predicate).await
    }

    async fn upsert_relationship(&self, rel: &RelationshipRecord) -> anyhow::Result<()> {
        Store::upsert_relationship(self, rel).await
    }

    async fn load_graph(&self) -> anyhow::Result<KnowledgeGraph> {
        Store::load_graph(self).await
    }
}
