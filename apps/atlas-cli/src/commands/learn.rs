//! `atlas learn` — durable learning (M07).
//!
//! input → document → chunks → embeddings → graph → SurrealDB.

use atlas_core::Config;

pub async fn run(config: &Config, text: &str) -> anyhow::Result<()> {
    // CLI-level guard: empty input is a usage error, not a document.
    // (The storage layer itself still accepts empty documents; that
    // behavior is pinned by storage tests and unchanged.)
    if text.trim().is_empty() {
        anyhow::bail!("usage: atlas learn <text> — the text must not be empty");
    }

    let store = atlas_storage::Store::open_durable(config.data_dir.join("surreal")).await?;
    let tokenizer = atlas_tokenizer::BpeTokenizer::embedded_cl100k();
    let embedder = atlas_embedding::FastembedEmbedder::bge_small_en_v1_5()?;

    let summary = atlas_storage::learn(&store, &tokenizer, &embedder, text).await?;
    if summary.duplicate {
        println!(
            "learn: duplicate document ({}, id {}) — nothing to do.",
            text.len(),
            summary.doc_id
        );
    } else {
        println!(
            "learn: stored {} (id {}) — {} chunks, {} embeddings, {} entities, {} relationships.",
            text.len(),
            summary.doc_id,
            summary.chunks,
            summary.embeddings,
            summary.entities,
            summary.relationships
        );
    }
    Ok(())
}
