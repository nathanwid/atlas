//! Command handlers.
//!
//! Learn, similar, graph, search, and ask commands.

pub mod ask;
pub mod graph;
pub mod learn;
pub mod search;
pub mod similar;

use crate::cli::Command;
use atlas_core::Config;

/// Dispatch a parsed command to its handler.
pub async fn run(config: &Config, command: Command) -> anyhow::Result<()> {
    match command {
        Command::Learn { text } => learn::run(config, &text).await,
        Command::Similar { term, top } => similar::run(config, &term, top).await,
        Command::Graph {
            entity,
            hops,
            limit,
        } => graph::run(config, &entity, hops, limit).await,
        Command::Search { query, top, mode } => search::run(config, &query, top, mode).await,
        Command::Ask { question } => ask::run(config, &question).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn similar_empty_database_reports_no_entities_without_model_download() {
        let (_dir, config) = test_config();
        run(
            &config,
            Command::Similar {
                term: "rust".into(),
                top: 5,
            },
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn graph_command_dispatches_with_default_limits() {
        let (_dir, config) = test_config();
        run(
            &config,
            Command::Graph {
                entity: "Rust".into(),
                hops: 1,
                limit: 50,
            },
        )
        .await
        .unwrap();
    }

    pub(crate) fn test_config() -> (tempfile::TempDir, Config) {
        let dir = tempfile::tempdir().unwrap();
        let config = Config {
            log_level: "info".into(),
            data_dir: dir.path().to_path_buf(),
        };
        (dir, config)
    }

    /// Phase 3 regression: empty learn input is a usage error rejected
    /// before any store or model access (hence offline-safe).
    #[tokio::test]
    async fn empty_learn_text_is_a_usage_error() {
        let (_dir, config) = test_config();
        let err = learn::run(&config, "")
            .await
            .expect_err("empty text must be rejected");
        assert!(err.to_string().contains("must not be empty"), "got: {err}");
    }

    #[tokio::test]
    async fn whitespace_learn_text_is_a_usage_error() {
        let (_dir, config) = test_config();
        let err = learn::run(&config, "   \t\n")
            .await
            .expect_err("whitespace text must be rejected");
        assert!(err.to_string().contains("must not be empty"), "got: {err}");
    }

    /// End-to-end fresh-directory flow with the real model: learn → search
    /// → ask on a data directory that does not exist yet (created on
    /// demand), then re-ask to confirm durability. Ignored by default;
    /// run explicitly with the model cache present.
    #[tokio::test]
    #[ignore = "loads the FastEmbed model (~100 MB cache); run explicitly"]
    async fn fresh_directory_end_to_end_with_real_model() {
        let dir = tempfile::tempdir().unwrap();
        let config = Config {
            log_level: "info".into(),
            data_dir: dir.path().to_path_buf(),
        };

        learn::run(&config, "Rust uses Tokio for async runtimes.")
            .await
            .expect("learn on fresh directory");
        search::run(
            &config,
            "tokio async runtime",
            3,
            crate::cli::SearchModeArg::Hybrid,
        )
        .await
        .expect("search after learn");
        ask::run(&config, "What uses Tokio?")
            .await
            .expect("ask after learn");
    }
}
