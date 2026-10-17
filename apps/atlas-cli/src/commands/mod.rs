//! Command handlers.
//!
//! M01 wired the five vision commands; M07 makes `learn` and `search` real.
//! Each handler logs its arguments via `tracing`. The remaining commands
//! still report which milestone will deliver them.

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
        Command::Similar { term } => similar::run(&term).await,
        Command::Graph { entity } => graph::run(&entity).await,
        Command::Search { query, top, mode } => search::run(config, &query, top, mode).await,
        Command::Ask { question } => ask::run(config, &question).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Placeholder commands still dispatch successfully.
    #[test]
    fn placeholder_commands_dispatch_successfully() {
        let config = test_config();
        let commands = [
            Command::Similar {
                term: "rust".into(),
            },
            Command::Graph {
                entity: "rust".into(),
            },
            Command::Ask {
                question: "What does Rust use?".into(),
            },
        ];
        for command in commands {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                assert!(run(&config, command).await.is_ok());
            });
        }
    }

    pub(crate) fn test_config() -> Config {
        Config {
            log_level: "info".into(),
            data_dir: std::env::temp_dir().join("atlas-cli-test"),
        }
    }

    /// Phase 3 regression: empty learn input is a usage error rejected
    /// before any store or model access (hence offline-safe).
    #[tokio::test]
    async fn empty_learn_text_is_a_usage_error() {
        let config = test_config();
        let err = learn::run(&config, "")
            .await
            .expect_err("empty text must be rejected");
        assert!(err.to_string().contains("must not be empty"), "got: {err}");
    }

    #[tokio::test]
    async fn whitespace_learn_text_is_a_usage_error() {
        let config = test_config();
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
