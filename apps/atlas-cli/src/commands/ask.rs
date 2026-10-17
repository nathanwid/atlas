//! `atlas ask` — question answering over Atlas's own knowledge (M09).
//!
//! question → hybrid retrieval → extractive answer with citations.

use atlas_core::Config;

pub async fn run(config: &Config, question: &str) -> anyhow::Result<()> {
    let store = atlas_storage::Store::open_durable(config.data_dir.join("surreal")).await?;
    let embedder = atlas_embedding::FastembedEmbedder::bge_small_en_v1_5()?;

    let answer = atlas_qa::ask(&store, &embedder, question).await?;
    println!("{}", answer.render());
    Ok(())
}
