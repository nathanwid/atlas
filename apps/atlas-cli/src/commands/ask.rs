//! `atlas ask` â€” question answering over Atlas's own knowledge (M09).
//!
//! question â†’ hybrid retrieval â†’ extractive answer with citations.

use atlas_core::Config;

pub async fn run(config: &Config, question: &str) -> anyhow::Result<()> {
    let embedder = atlas_embedding::FastembedEmbedder::bge_small_en_v1_5()?;

    println!(
        "{}",
        answer_with_embedder(config, &embedder, question).await?
    );
    Ok(())
}

async fn answer_with_embedder(
    config: &Config,
    embedder: &dyn atlas_embedding::Embedder,
    question: &str,
) -> anyhow::Result<String> {
    let store = atlas_storage::SqliteStore::open_durable(config.data_dir.join("atlas.db")).await?;
    Ok(atlas_qa::ask(&store, embedder, question).await?.render())
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fake;
    impl atlas_embedding::Embedder for Fake {
        fn model_id(&self) -> &str {
            "cli-test"
        }
        fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0]).collect())
        }
    }
    #[tokio::test]
    async fn fake_backed_ask_is_grounded_and_isolated() {
        let (_dir, config) = crate::commands::tests::test_config();
        let (_other_dir, other) = crate::commands::tests::test_config();
        assert_ne!(config.data_dir, other.data_dir);
        let store = atlas_storage::SqliteStore::open_durable(config.data_dir.join("atlas.db"))
            .await
            .unwrap();
        let tok = atlas_tokenizer::BpeTokenizer::embedded_cl100k();
        atlas_storage::learn(&store, &tok, &Fake, "Rust uses Tokio.")
            .await
            .unwrap();
        let answer = answer_with_embedder(&config, &Fake, "Rust uses Tokio")
            .await
            .unwrap();
        assert!(answer.contains("Rust uses Tokio"));
        assert!(answer.contains("chunk"));
        assert!(
            answer_with_embedder(&other, &Fake, "Rust")
                .await
                .unwrap()
                .contains("No sufficient evidence")
        );
    }
}
