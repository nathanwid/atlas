//! The durable `learn` and `search` workflows.
//!
//! `learn`: `input → document → chunks → embeddings → graph → persistence`.
//! SQLite writes the complete record group in one transaction. Re-running
//! `learn` repairs missing derived records and keeps existing graph evidence.
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
/// Generic over the storage [`Backend`]; today that is [`SqliteStore`].
pub async fn learn(
    store: &impl Backend,
    tokenizer: &BpeTokenizer,
    embedder: &dyn Embedder,
    text: &str,
) -> anyhow::Result<LearnSummary> {
    let doc_id = doc_id_for(text);
    let chunker = Chunker::new(tokenizer, 128, 0);
    let chunks = chunker.chunk(text);
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
    let extractor = Extractor::new(ExtractorConfig::default());
    let mut graph = KnowledgeGraph::new();
    for record in &chunk_records {
        extractor.extract(&record.chunk_id, &record.text, &mut graph);
    }
    let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text).collect();
    embedder.validate_documents(&chunk_texts)?;
    if store
        .learn_is_complete(&doc_id, &model_id, &chunk_records, &graph)
        .await?
    {
        return Ok(LearnSummary {
            doc_id,
            duplicate: true,
            chunks: 0,
            embeddings: 0,
            entities: 0,
            relationships: 0,
        });
    }
    let vectors = embedder.embed_documents(&chunk_texts)?;
    anyhow::ensure!(
        vectors.len() == chunks.len(),
        "embedder returned wrong vector count"
    );
    atlas_embedding::validate_vectors(&vectors)?;
    let embedding_records: Vec<EmbeddingRecord> = chunk_records
        .iter()
        .zip(vectors)
        .map(|(record, vector)| EmbeddingRecord {
            chunk_id: record.chunk_id.clone(),
            model_id: model_id.clone(),
            vector,
        })
        .collect();

    // Persist all derived records in the existing atomic group.
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
    if k == 0 {
        return Ok(Vec::new());
    }
    let model_id = embedder.model_id();
    let embeddings = store.load_embeddings(model_id).await?;
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    let dim = embeddings[0].vector.len();
    anyhow::ensure!(dim > 0, "index dimension must be positive");
    // Keep records for chunk IDs; do not clone another full vector batch.
    let mut index = crate::brute::BruteForceIndex::with_capacity(dim, embeddings.len());
    for embedding in &embeddings {
        index.add(&embedding.vector)?;
    }
    let query_vector = embedder.embed_query(query)?;
    anyhow::ensure!(
        query_vector.len() == dim && query_vector.iter().all(|x| x.is_finite()),
        "query embedding must have {dim} finite components"
    );

    let mut results = Vec::new();
    for hit in crate::index::VectorIndex::search(&index, &query_vector, embeddings.len()) {
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
        if results.len() == k {
            break;
        }
    }
    Ok(results)
}
