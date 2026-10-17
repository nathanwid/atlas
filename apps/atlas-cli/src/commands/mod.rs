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
        Command::Ask { question } => ask::run(&question).await,
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
}
