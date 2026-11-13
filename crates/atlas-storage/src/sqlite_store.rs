//! SQLite store: the production backend for Atlas.
//!
//! Seven relational tables mirror the record shapes: `document`, `chunk`,
//! `embedding`, `entity`, `predicate`, `relationship`, and `evidence`
//! (normalized out of the relationship record so a relationship's evidence
//! set is replaceable as a unit). The record structs are the schema; here
//! `document`/`chunk`/`entity`/`predicate` rows carry the full record as
//! JSON, `embedding` and `relationship`/`evidence` use explicit columns
//! for the fields other queries touch.
//!
//! Durability configuration (per SQLite's documented WAL contract):
//! `journal_mode=WAL` (persistent per database file) plus
//! `synchronous=FULL` on every connection, which syncs the WAL on every
//! transaction commit — durable across application crash and, per the
//! upstream documentation, power loss. `busy_timeout` guards against
//! `SQLITE_BUSY` from a concurrent connection (e.g. the crash test).
//!
//! Concurrency: rusqlite's `Connection` is `Send` but not `Sync`, so it
//! lives behind a `Mutex`. All SQLite calls are synchronous and short; no
//! await happens while the lock is held. `spawn_blocking` is deliberately
//! absent until benches show blocking to matter.
//!
//! Read ordering: `load_embeddings`/`load_graph` read rows in insertion
//! (rowid) order and sort in Rust by ordinal.

use crate::backend::Backend;
use crate::records::{
    ChunkRecord, EmbeddingRecord, EntityRecord, EvidenceRecord, PredicateRecord, RelationshipRecord,
};
use atlas_graph::{Evidence, KnowledgeGraph};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

pub struct SqliteStore {
    conn: Mutex<Connection>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS document (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunk (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embedding (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS entity (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predicate (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS relationship (
    id TEXT PRIMARY KEY,
    src TEXT NOT NULL,
    predicate TEXT NOT NULL,
    dst TEXT NOT NULL,
    origin TEXT NOT NULL,
    ordinal INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    relationship_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    sentence TEXT NOT NULL,
    PRIMARY KEY (relationship_id, ordinal)
);
";

const BUSY_TIMEOUT_MS: u64 = 5000;

impl SqliteStore {
    /// In-memory SQLite store for tests.
    pub async fn open_mem() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn, false)
    }

    /// Durable store at `path`. SurrealDB's old `Store::open_durable` took
    /// directory; to keep call sites compatible, a directory argument gets
    /// an `atlas.db` file inside it, and a file argument is used as-is.
    pub async fn open_durable(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let file = if path.is_dir() {
            path.join("atlas.db")
        } else {
            path.to_path_buf()
        };
        if let Some(parent) = file.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&file)?;
        Self::init(conn, true)
    }

    fn init(conn: Connection, durable: bool) -> anyhow::Result<Self> {
        conn.busy_timeout(Duration::from_millis(BUSY_TIMEOUT_MS))?;
        if durable {
            let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
            anyhow::ensure!(
                mode == "wal",
                "WAL mode could not be activated, journal_mode is {mode:?}"
            );
        }
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(SqliteStore {
            conn: Mutex::new(conn),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().expect("sqlite connection mutex poisoned")
    }

    fn upsert_json<T: serde::Serialize>(
        &self,
        table: &str,
        id: &str,
        value: &T,
    ) -> anyhow::Result<()> {
        let conn = self.lock();
        upsert_json_on(&conn, table, id, value)
    }
}

/// Free functions over a bare `&Connection` so the same write logic runs
/// through the autocommit primitives and inside the grouped
/// `apply_learn_write` transaction (`Transaction` derefs to `Connection`).
fn upsert_json_on<T: serde::Serialize>(
    conn: &Connection,
    table: &str,
    id: &str,
    value: &T,
) -> anyhow::Result<()> {
    let json = serde_json::to_string(value)?;
    let n = conn.execute(
        &format!(
            "INSERT INTO {table}(id, value) VALUES (?1, ?2)
             ON CONFLICT(id) DO UPDATE SET value = excluded.value"
        ),
        params![id, json],
    )?;
    debug_assert_eq!(n, 1);
    Ok(())
}

fn record_exists(
    conn: &Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> anyhow::Result<bool> {
    Ok(conn
        .query_row(sql, params, |_| Ok(()))
        .optional()?
        .is_some())
}

fn next_json_ordinal(conn: &Connection, table: &str) -> anyhow::Result<u32> {
    let next: i64 = conn.query_row(
        &format!("SELECT COALESCE(MAX(json_extract(value, '$.ordinal')), -1) + 1 FROM {table}"),
        [],
        |r| r.get(0),
    )?;
    Ok(next.try_into()?)
}

/// Upsert one relationship row and replace its whole evidence set. No
/// transaction of its own: callers decide the transaction scope.
fn write_relationship_on(conn: &Connection, rel: &RelationshipRecord) -> anyhow::Result<()> {
    let id = crate::records::relationship_id(&rel.src, &rel.predicate, &rel.dst);
    conn.execute(
        "INSERT INTO relationship(id, src, predicate, dst, origin, ordinal)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(id) DO UPDATE SET
             src = excluded.src,
             predicate = excluded.predicate,
             dst = excluded.dst,
             origin = excluded.origin,
             ordinal = excluded.ordinal",
        params![id, rel.src, rel.predicate, rel.dst, rel.origin, rel.ordinal],
    )?;
    // The evidence set is part of the relationship's identity: replace
    // it wholesale so an upsert is idempotent under re-learning.
    conn.execute(
        "DELETE FROM evidence WHERE relationship_id = ?1",
        params![id],
    )?;
    for (ordinal, ev) in rel.evidence.iter().enumerate() {
        conn.execute(
            "INSERT INTO evidence(relationship_id, ordinal, doc_id, sentence)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, ordinal as i64, ev.doc_id, ev.sentence],
        )?;
    }
    Ok(())
}

/// Merge a document's evidence into an existing relationship. The caller
/// must hold the write transaction before these reads, so another writer
/// cannot replace the evidence between the read and the write.
fn merge_learn_relationship_on(conn: &Connection, rel: &RelationshipRecord) -> anyhow::Result<()> {
    let id = crate::records::relationship_id(&rel.src, &rel.predicate, &rel.dst);
    let previous: Option<(String, u32)> = conn
        .query_row(
            "SELECT origin, ordinal FROM relationship WHERE id = ?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((origin, ordinal)) = previous else {
        let mut rel = rel.clone();
        let next: i64 = conn.query_row(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM relationship",
            [],
            |r| r.get(0),
        )?;
        rel.ordinal = next.try_into()?;
        return write_relationship_on(conn, &rel);
    };

    let mut merged = rel.clone();
    merged.ordinal = ordinal;
    if origin == "explicit" {
        merged.origin = origin;
    }
    let mut stmt = conn.prepare(
        "SELECT doc_id, sentence FROM evidence WHERE relationship_id = ?1 ORDER BY ordinal",
    )?;
    let mut evidence = stmt
        .query_map(params![id], |row| {
            Ok(EvidenceRecord {
                doc_id: row.get(0)?,
                sentence: row.get(1)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    for ev in &rel.evidence {
        if !evidence
            .iter()
            .any(|old| old.doc_id == ev.doc_id && old.sentence == ev.sentence)
        {
            evidence.push(ev.clone());
        }
    }
    merged.evidence = evidence;
    write_relationship_on(conn, &merged)
}

fn write_embedding_on(conn: &Connection, embedding: &EmbeddingRecord) -> anyhow::Result<()> {
    // Embedding key is (chunk_id, model_id); record id joins both.
    let id = format!("{}:{}", embedding.model_id, embedding.chunk_id);
    conn.execute(
        "INSERT INTO embedding(id, model_id, chunk_id, vector) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(id) DO UPDATE SET
             model_id = excluded.model_id,
             chunk_id = excluded.chunk_id,
             vector = excluded.vector",
        params![
            id,
            embedding.model_id,
            embedding.chunk_id,
            vector_to_blob(&embedding.vector)
        ],
    )?;
    Ok(())
}

fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * 4);
    for f in vector {
        blob.extend_from_slice(&f.to_le_bytes());
    }
    blob
}

fn blob_to_vector(blob: &[u8]) -> anyhow::Result<Vec<f32>> {
    // `ensure!` keeps trailing bytes an error, so the `as_chunks`
    // remainder is always empty here.
    anyhow::ensure!(
        blob.len().is_multiple_of(4),
        "embedding blob length {} is not a multiple of 4",
        blob.len()
    );
    let (chunks, remainder) = blob.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

impl Backend for SqliteStore {
    async fn upsert_document(&self, doc_id: &str, text: &str) -> anyhow::Result<()> {
        self.upsert_json(
            "document",
            doc_id,
            &crate::records::DocumentRecord {
                doc_id: doc_id.to_string(),
                text: text.to_string(),
            },
        )
    }

    async fn document_exists(&self, doc_id: &str) -> anyhow::Result<bool> {
        let found: Option<i64> = self
            .lock()
            .query_row(
                "SELECT 1 FROM document WHERE id = ?1",
                params![doc_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(found.is_some())
    }

    async fn upsert_chunk(&self, chunk: &ChunkRecord) -> anyhow::Result<()> {
        self.upsert_json("chunk", &chunk.chunk_id, chunk)
    }

    async fn get_chunk(&self, chunk_id: &str) -> anyhow::Result<Option<ChunkRecord>> {
        let json: Option<String> = self
            .lock()
            .query_row(
                "SELECT value FROM chunk WHERE id = ?1",
                params![chunk_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(json.map(|j| serde_json::from_str(&j)).transpose()?)
    }

    async fn delete_chunk(&self, chunk_id: &str) -> anyhow::Result<()> {
        self.lock()
            .execute("DELETE FROM chunk WHERE id = ?1", params![chunk_id])?;
        Ok(())
    }

    async fn upsert_embedding(&self, embedding: &EmbeddingRecord) -> anyhow::Result<()> {
        let conn = self.lock();
        write_embedding_on(&conn, embedding)
    }

    async fn load_embeddings(&self, model_id: &str) -> anyhow::Result<Vec<EmbeddingRecord>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT model_id, chunk_id, vector FROM embedding
             WHERE model_id = ?1 ORDER BY rowid",
        )?;
        let rows = stmt.query_map(params![model_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (model_id, chunk_id, blob) = row?;
            out.push(EmbeddingRecord {
                chunk_id,
                model_id,
                vector: blob_to_vector(&blob)?,
            });
        }
        Ok(out)
    }

    async fn upsert_entity(&self, entity: &EntityRecord) -> anyhow::Result<()> {
        self.upsert_json("entity", &entity.key, entity)
    }

    async fn upsert_predicate(&self, predicate: &PredicateRecord) -> anyhow::Result<()> {
        self.upsert_json("predicate", &predicate.key, predicate)
    }

    async fn upsert_relationship(&self, rel: &RelationshipRecord) -> anyhow::Result<()> {
        let mut conn = self.lock();
        // A relationship and its evidence are one logical row group: keep
        // their rewrite atomic in the autocommit path too.
        let tx = conn.transaction()?;
        write_relationship_on(&tx, rel)?;
        tx.commit()?;
        Ok(())
    }

    async fn load_graph(&self) -> anyhow::Result<KnowledgeGraph> {
        let mut guard = self.lock();
        let tx = guard.transaction()?;
        let conn = &tx;

        let mut entities: Vec<EntityRecord> = {
            let mut stmt = conn.prepare("SELECT value FROM entity ORDER BY rowid")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<Result<Vec<_>, _>>()?
                .iter()
                .map(|j| serde_json::from_str(j))
                .collect::<Result<Vec<_>, _>>()?
        };
        #[cfg(test)]
        AFTER_ENTITIES.with(|hook| {
            if let Some(hook) = hook.borrow_mut().take() {
                hook();
            }
        });
        let mut predicates: Vec<PredicateRecord> = {
            let mut stmt = conn.prepare("SELECT value FROM predicate ORDER BY rowid")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<Result<Vec<_>, _>>()?
                .iter()
                .map(|j| serde_json::from_str(j))
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut relationships: Vec<(String, RelationshipRecord)> = {
            let mut stmt = conn.prepare(
                "SELECT id, src, predicate, dst, origin, ordinal
                 FROM relationship ORDER BY rowid",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, i64>(5)?,
                ))
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(
                    |(id, src, predicate, dst, origin, ordinal)| -> Result<_, serde_json::Error> {
                        Ok((
                            id,
                            RelationshipRecord {
                                src,
                                predicate,
                                dst,
                                origin,
                                evidence: Vec::new(),
                                ordinal: ordinal as u32,
                            },
                        ))
                    },
                )
                .collect::<Result<Vec<_>, _>>()?
        };
        // Evidence rows in (relationship, insertion) order, grouped by id.
        let mut evidence: std::collections::HashMap<String, Vec<EvidenceRecord>> =
            std::collections::HashMap::new();
        {
            let mut stmt = conn.prepare(
                "SELECT relationship_id, doc_id, sentence FROM evidence
                 ORDER BY relationship_id, ordinal",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?;
            for row in rows {
                let (id, doc_id, sentence) = row?;
                evidence
                    .entry(id)
                    .or_default()
                    .push(EvidenceRecord { doc_id, sentence });
            }
        }
        tx.commit()?;
        drop(guard);

        entities.sort_by_key(|e| e.ordinal);
        predicates.sort_by_key(|p| p.ordinal);
        relationships.sort_by_key(|(_, r)| r.ordinal);

        let mut graph = KnowledgeGraph::new();
        for entity in &entities {
            let id = graph.intern_entity(&entity.display);
            if let Some(kind) = &entity.kind {
                graph.set_kind(id, kind);
            }
        }
        for predicate in &predicates {
            graph.intern_predicate(&predicate.key);
        }
        for (id, rel) in &relationships {
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
            for ev in evidence.get(id).map(Vec::as_slice).unwrap_or(&[]) {
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

    async fn learn_is_complete(
        &self,
        doc_id: &str,
        model_id: &str,
        chunks: &[ChunkRecord],
        graph: &KnowledgeGraph,
    ) -> anyhow::Result<bool> {
        let mut guard = self.lock();
        let tx = guard.transaction()?;
        if !record_exists(&tx, "SELECT 1 FROM document WHERE id=?1", params![doc_id])? {
            return Ok(false);
        }
        for expected in chunks {
            let json: Option<String> = tx
                .query_row(
                    "SELECT value FROM chunk WHERE id=?1",
                    params![expected.chunk_id],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(json) = json else {
                return Ok(false);
            };
            let actual: ChunkRecord = serde_json::from_str(&json)?;
            if actual.text != expected.text
                || actual.doc_id != expected.doc_id
                || actual.ordinal != expected.ordinal
            {
                return Ok(false);
            }
            if !record_exists(
                &tx,
                "SELECT 1 FROM embedding WHERE id=?1 AND model_id=?2 AND chunk_id=?3",
                params![
                    format!("{model_id}:{}", expected.chunk_id),
                    model_id,
                    expected.chunk_id
                ],
            )? {
                return Ok(false);
            }
        }
        for (id, _) in graph.entities() {
            if !record_exists(
                &tx,
                "SELECT 1 FROM entity WHERE id=?1",
                params![graph.entity_key(id).unwrap()],
            )? {
                return Ok(false);
            }
        }
        for (_, key) in graph.predicates() {
            if !record_exists(&tx, "SELECT 1 FROM predicate WHERE id=?1", params![key])? {
                return Ok(false);
            }
        }
        for (_, rel) in graph.relationships() {
            let id = crate::records::relationship_id(
                &graph.entity_key(rel.src).unwrap(),
                graph.predicate(rel.predicate).unwrap(),
                &graph.entity_key(rel.dst).unwrap(),
            );
            let origin: Option<String> = tx
                .query_row(
                    "SELECT origin FROM relationship WHERE id=?1",
                    params![id],
                    |r| r.get(0),
                )
                .optional()?;
            if origin.is_none()
                || (rel.origin == atlas_graph::EdgeOrigin::Explicit
                    && origin.as_deref() != Some("explicit"))
            {
                return Ok(false);
            }
            for ev in &rel.evidence {
                if !record_exists(
                    &tx,
                    "SELECT 1 FROM evidence WHERE relationship_id=?1 AND doc_id=?2 AND sentence=?3",
                    params![id, ev.doc_id, ev.sentence],
                )? {
                    return Ok(false);
                }
            }
        }
        tx.commit()?;
        Ok(true)
    }

    /// Grouped write: document, chunks, embeddings, and the whole graph in
    /// ONE transaction — one WAL sync instead of one per record. Any error
    /// rolls the entire group back (the `Transaction` drops unresolved),
    /// so a failed `learn` cannot leave a partial subset of its writes.
    /// The write order is document, chunks, embeddings, then graph records.
    async fn apply_learn_write(
        &self,
        write: &crate::backend::LearnWrite<'_>,
    ) -> anyhow::Result<()> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        // Acquire the writer lock before reading evidence for the merge.
        upsert_json_on(&tx, "document", &write.document.doc_id, write.document)?;
        for chunk in write.chunks {
            let previous: Option<String> = tx
                .query_row(
                    "SELECT value FROM chunk WHERE id=?1",
                    params![chunk.chunk_id],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(previous) = previous {
                let previous: ChunkRecord = serde_json::from_str(&previous)?;
                anyhow::ensure!(
                    previous.text == chunk.text
                        && previous.doc_id == chunk.doc_id
                        && previous.ordinal == chunk.ordinal,
                    "chunk {} already has different text or layout; keep the original tokenizer and chunk settings, or rebuild into a new data directory to change the layout",
                    chunk.chunk_id
                );
            }
            upsert_json_on(&tx, "chunk", &chunk.chunk_id, chunk)?;
        }
        for embedding in write.embeddings {
            write_embedding_on(&tx, embedding)?;
        }
        for (_, entity) in write.graph.entities() {
            let key = atlas_graph::normalize(&entity.display);
            if record_exists(&tx, "SELECT 1 FROM entity WHERE id=?1", params![key])? {
                continue;
            }
            let ordinal: u32 = next_json_ordinal(&tx, "entity")?;
            upsert_json_on(
                &tx,
                "entity",
                &atlas_graph::normalize(&entity.display),
                &EntityRecord {
                    key: atlas_graph::normalize(&entity.display),
                    display: entity.display.clone(),
                    kind: entity.kind.clone(),
                    ordinal,
                },
            )?;
        }
        for (_, key) in write.graph.predicates() {
            if record_exists(&tx, "SELECT 1 FROM predicate WHERE id=?1", params![key])? {
                continue;
            }
            let ordinal: u32 = next_json_ordinal(&tx, "predicate")?;
            upsert_json_on(
                &tx,
                "predicate",
                key,
                &PredicateRecord {
                    key: key.to_string(),
                    ordinal,
                },
            )?;
        }
        for (ordinal, (_, rel)) in write.graph.relationships().enumerate() {
            merge_learn_relationship_on(
                &tx,
                &RelationshipRecord {
                    src: write.graph.entity_key(rel.src).ok_or_else(|| {
                        anyhow::anyhow!("relationship references unknown source entity")
                    })?,
                    predicate: write
                        .graph
                        .predicate(rel.predicate)
                        .unwrap_or_default()
                        .to_string(),
                    dst: write.graph.entity_key(rel.dst).ok_or_else(|| {
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
                },
            )?;
        }
        #[cfg(test)]
        BEFORE_COMMIT.with(|hook| {
            if let Some(hook) = hook.borrow().as_ref() {
                hook();
            }
        });
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    static AFTER_ENTITIES: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = Default::default();
    static BEFORE_COMMIT: std::cell::RefCell<Option<Box<dyn Fn()>>> = Default::default();
}

#[cfg(test)]
mod crash_tests {
    use super::*;
    use crate::backend::LearnWrite;
    use crate::records::DocumentRecord;

    async fn apply_source(store: &SqliteStore, id: &str, extend: bool) {
        let mut graph = KnowledgeGraph::new();
        let rust = graph.intern_entity("Rust");
        let tokio = graph.intern_entity("Tokio");
        let uses = graph.intern_predicate("uses");
        let sentence = format!("{id} source");
        let evidence = Evidence {
            doc_id: format!("{id}:0"),
            sentence: sentence.clone(),
        };
        graph.link(
            rust,
            uses,
            tokio,
            atlas_graph::EdgeOrigin::Explicit,
            evidence.clone(),
        );
        if extend {
            let cargo = graph.intern_entity("Cargo");
            let requires = graph.intern_predicate("requires");
            graph.link(
                tokio,
                requires,
                cargo,
                atlas_graph::EdgeOrigin::Explicit,
                evidence,
            );
        }
        store
            .apply_learn_write(&LearnWrite {
                document: &DocumentRecord {
                    doc_id: id.into(),
                    text: sentence.clone(),
                },
                chunks: &[ChunkRecord {
                    chunk_id: format!("{id}:0"),
                    doc_id: id.into(),
                    ordinal: 0,
                    text: sentence,
                }],
                embeddings: &[EmbeddingRecord {
                    chunk_id: format!("{id}:0"),
                    model_id: "crash".into(),
                    vector: vec![1.0],
                }],
                graph: &graph,
            })
            .await
            .unwrap();
    }

    #[test]
    fn child() {
        let Some(db) = std::env::var_os("ATLAS_UNIT_CRASH_DB") else {
            return;
        };
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let store = SqliteStore::open_durable(db).await.unwrap();
            BEFORE_COMMIT.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(|| {
                    std::fs::write(
                        std::env::var_os("ATLAS_UNIT_CRASH_MARKER").unwrap(),
                        b"open",
                    )
                    .unwrap();
                    loop {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }))
            });
            apply_source(&store, "new", true).await;
        });
    }

    #[test]
    fn kill_during_group_rolls_back_and_keeps_old_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("atlas.db");
        let marker = dir.path().join("ready");
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let store = SqliteStore::open_durable(&db).await.unwrap();
            apply_source(&store, "old", false).await;
        });
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "sqlite_store::crash_tests::child"])
            .env("ATLAS_UNIT_CRASH_DB", &db)
            .env("ATLAS_UNIT_CRASH_MARKER", &marker)
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while !marker.exists() && std::time::Instant::now() < deadline {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let ready = marker.exists();
        let _ = child.kill();
        child.wait().unwrap();
        assert!(ready, "child did not reach transaction pause");
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let store = SqliteStore::open_durable(&db).await.unwrap();
            assert!(!store.document_exists("new").await.unwrap());
            let graph = store.load_graph().await.unwrap();
            assert_eq!(graph.relationship_count(), 1);
            assert_eq!(
                graph.relationships().next().unwrap().1.evidence[0].sentence,
                "old source"
            );
            let conn = store.lock();
            for (table, expected) in [
                ("document", 1),
                ("chunk", 1),
                ("embedding", 1),
                ("entity", 2),
                ("predicate", 1),
                ("relationship", 1),
                ("evidence", 1),
            ] {
                let count: u32 = conn
                    .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                    .unwrap();
                assert_eq!(count, expected, "rolled back {table}");
            }
            let integrity: String = conn
                .query_row("PRAGMA integrity_check", [], |r| r.get(0))
                .unwrap();
            assert_eq!(integrity, "ok");
        });
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    #[tokio::test]
    async fn graph_load_uses_one_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atlas.db");
        let store = SqliteStore::open_durable(&path).await.unwrap();
        store
            .upsert_entity(&EntityRecord {
                key: "a".into(),
                display: "A".into(),
                kind: None,
                ordinal: 0,
            })
            .await
            .unwrap();
        AFTER_ENTITIES.with(|hook| *hook.borrow_mut()=Some(Box::new(move|| {
   std::thread::spawn(move|| {
    let conn=Connection::open(path).unwrap();
    conn.execute_batch(r#"BEGIN; INSERT INTO entity VALUES ('b','{"key":"b","display":"B","kind":null,"ordinal":1}');
    INSERT INTO predicate VALUES ('uses','{"key":"uses","ordinal":0}');
    INSERT INTO relationship VALUES ('r','a','uses','b','explicit',0);
    INSERT INTO evidence VALUES ('r',0,'doc:0','A uses B'); COMMIT;"#).unwrap();
   }).join().unwrap();
  })));
        assert_eq!(store.load_graph().await.unwrap().relationship_count(), 0);
        assert_eq!(store.load_graph().await.unwrap().relationship_count(), 1);
    }
}
