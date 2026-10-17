//! `atlas search` — search persisted content (M07 vector, M08 hybrid).
//!
//! query → embed → persisted embeddings / graph expansion → ranked chunks.

use atlas_core::Config;

pub async fn run(
    config: &Config,
    query: &str,
    k: usize,
    mode: crate::cli::SearchModeArg,
) -> anyhow::Result<()> {
    let store = atlas_storage::Store::open_durable(config.data_dir.join("surreal")).await?;
    let embedder = atlas_embedding::FastembedEmbedder::bge_small_en_v1_5()?;

    let mode = match mode {
        crate::cli::SearchModeArg::Vector => atlas_storage::SearchMode::Vector,
        crate::cli::SearchModeArg::Hybrid => atlas_storage::SearchMode::Hybrid,
        crate::cli::SearchModeArg::Graph => atlas_storage::SearchMode::Graph,
    };
    let results =
        atlas_storage::hybrid_search(&store, &embedder, query, k, &Default::default(), mode)
            .await?;

    if results.is_empty() {
        if mode == atlas_storage::SearchMode::Graph {
            println!("search: no graph-expanded results for this query.");
        } else {
            println!("search: nothing learned yet — run `atlas learn <text>` first.");
        }
        return Ok(());
    }
    for (rank, hit) in results.iter().enumerate() {
        let score = hit
            .score
            .map(|s| format!("{s:.4}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "#{rank} source={} score={score} chunk={} doc={}",
            hit.source, hit.chunk_id, hit.doc_id
        );
        for via in &hit.via {
            println!("  via {via}");
        }
        println!("  {}", hit.text.replace('\n', " "));
    }
    Ok(())
}
