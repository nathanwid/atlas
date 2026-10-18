//! The durable `learn` and `search` workflows.
//!
//! `learn`: `input → document → chunks → embeddings → graph → persistence`.
//! Every write is an idempotent upsert keyed by deterministic IDs, so a
//! duplicate `learn` is a no-op and a partial failure is repaired by simply
//! running `learn` again — no transactions needed.
//!
//! `search`: `query → embed_query → load persisted embeddings →
//! BruteForceIndex → top-k chunks`. The index is rebuilt from the
//! `embedding` table on demand; at Atlas's scale a full exact scan is the
//! right answer, and it keeps the [`VectorIndex`] abstraction as the single
//! search surface.
//!
//! Graph extraction runs per chunk with `doc_id = chunk_id`, so every
//! relationship's evidence points at the chunk that produced it — the hook
//! M08 hybrid retrieval will walk.

use crate::backend::Backend;
use crate::backend::LearnWrite;
use crate::records::{ChunkRecord, DocumentRecord, EmbeddingRecord, doc_id_for};
use atlas_embedding::{Chunker, Embedder};
use atlas_graph::{Extractor, ExtractorConfig, KnowledgeGraph};
use atlas_tokenizer::BpeTokenizer;

/// What one `learn` call changed. For a duplicate document every count is
/// zero and `duplicate` is true.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnSummary {
    pub doc_id: String,
    pub duplicate: bool,
    pub chunks: usize,
    pub embeddings: usize,
    pub entities: usize,
    pub relationships: usize,
}

/// Learn from a piece of text: chunk, embed, extract, persist.
///
/// Generic over the storage [`Backend`], so the SurrealDB `Store` and the
/// opt-in SQLite backend run the identical pipeline (PoC parity guarantee).
pub async fn learn(
    store: &impl Backend,
    tokenizer: &BpeTokenizer,
    embedder: &dyn Embedder,
    text: &str,
) -> anyhow::Result<LearnSummary> {
    let doc_id = doc_id_for(text);
    let chunker = Chunker::new(tokenizer, 128, 0);
    let chunks = chunker.chunk(text);
    let chunk_ids: Vec<String> = chunks
        .iter()
        .enumerate()
        .map(|(ordinal, _)| format!("{doc_id}:{ordinal}"))
        .collect();

    // Duplicate detection doubles as partial-failure repair: the durable
    // learn flow is a sequence of idempotent upserts with no transaction, so
    // an interrupted run can leave the document row without all of its
    // chunks/embeddings. Only a document whose *complete* chunk set exists
    // counts as a duplicate; a partial one falls through and is repaired by
    // re-running the idempotent pipeline. (A crash mid-chunk-write can
    // still leave a gap between two checks — repaired the same way on the
    // next learn.)
    if store.document_exists(&doc_id).await? {
        let mut complete = true;
        for id in &chunk_ids {
            if store.get_chunk(id).await?.is_none() {
                complete = false;
                break;
            }
        }
        if complete {
            return Ok(LearnSummary {
                doc_id,
                duplicate: true,
                chunks: 0,
                embeddings: 0,
                entities: 0,
                relationships: 0,
            });
        }
    }

    let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text).collect();

    // One batched embedding call for the whole document.
    let vectors = embedder.embed_documents(&chunk_texts)?;
    anyhow::ensure!(
        vectors.len() == chunks.len(),
        "embedder returned wrong vector count"
    );

    let model_id = embedder.model_id().to_string();
    let chunk_records: Vec<ChunkRecord> = chunks
        .iter()
        .enumerate()
        .map(|(ordinal, chunk)| ChunkRecord {
            chunk_id: format!("{doc_id}:{ordinal}"),
            doc_id: doc_id.clone(),
            ordinal: ordinal as u32,
            text: chunk.text.to_string(),
        })
        .collect();
    let embedding_records: Vec<EmbeddingRecord> = chunk_records
        .iter()
        .zip(vectors)
        .map(|(record, vector)| EmbeddingRecord {
            chunk_id: record.chunk_id.clone(),
            model_id: model_id.clone(),
            vector,
        })
        .collect();

    // Graph extraction runs per chunk so evidence points at chunks. The
    // chunk id here is exactly the one persisted below.
    let extractor = Extractor::new(ExtractorConfig::default());
    let mut graph = KnowledgeGraph::new();
    for record in &chunk_records {
        extractor.extract(&record.chunk_id, &record.text, &mut graph);
    }

    // A single grouped write: transactional backends commit document,
    // chunks, embeddings, and graph atomically; others fall back to the
    // sequential primitive writes (same order as before batching).
    let document = DocumentRecord {
        doc_id: doc_id.clone(),
        text: text.to_string(),
    };
    store
        .apply_learn_write(&LearnWrite {
            document: &document,
            chunks: &chunk_records,
            embeddings: &embedding_records,
            graph: &graph,
        })
        .await?;

    let summary = LearnSummary {
        doc_id,
        duplicate: false,
        chunks: chunks.len(),
        embeddings: embedding_records.len(),
        entities: graph.entity_count(),
        relationships: graph.relationship_count(),
    };
    Ok(summary)
}

/// One search hit: the chunk, its document, and its similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub chunk_id: String,
    pub doc_id: String,
    pub score: f32,
    pub text: String,
}

/// Search persisted embeddings: rebuild the exact index from stored
/// vectors, embed the query, rank, and hydrate chunk text.
pub async fn search(
    store: &impl Backend,
    embedder: &dyn Embedder,
    query: &str,
    k: usize,
) -> anyhow::Result<Vec<SearchResult>> {
    let model_id = embedder.model_id();
    let embeddings = store.load_embeddings(model_id).await?;
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    let dim = embeddings[0].vector.len();
    let vectors: Vec<Vec<f32>> = embeddings.iter().map(|e| e.vector.clone()).collect();
    let index = crate::brute::BruteForceIndex::build(dim, &vectors)?;
    let query_vector = embedder.embed_query(query)?;

    let mut results = Vec::new();
    for hit in crate::index::VectorIndex::search(&index, &query_vector, k) {
        let chunk_id = &embeddings[hit.id as usize].chunk_id;
        let Some(chunk) = store.get_chunk(chunk_id).await? else {
            // Orphaned embedding (its chunk was deleted independently).
            // Skip rather than fail the whole query: partial states must
            // degrade, not break retrieval. Re-learn repairs the row.
            continue;
        };
        results.push(SearchResult {
            chunk_id: chunk.chunk_id,
            doc_id: chunk.doc_id,
            score: hit.score,
            text: chunk.text,
        });
    }
    Ok(results)
}
