//! `atlas search` — search persisted embeddings (M07).
//!
//! query → embed_query → persisted embeddings → BruteForceIndex → chunks.

use atlas_core::Config;

pub async fn run(config: &Config, query: &str, k: usize) -> anyhow::Result<()> {
    let store = atlas_storage::Store::open_durable(config.data_dir.join("surreal")).await?;
    let embedder = atlas_embedding::FastembedEmbedder::bge_small_en_v1_5()?;

    let results = atlas_storage::search(&store, &embedder, query, k).await?;
    if results.is_empty() {
        println!("search: nothing learned yet — run `atlas learn <text>` first.");
        return Ok(());
    }
    for (rank, hit) in results.iter().enumerate() {
        println!(
            "#{rank} score={:.4} chunk={} doc={}",
            hit.score, hit.chunk_id, hit.doc_id
        );
        println!("  {}", hit.text.replace('\n', " "));
    }
    Ok(())
}
