//! Command handlers.
//!
//! M01 wires the five vision commands end to end: each handler logs its
//! arguments and reports which milestone will deliver its functionality.
//! Behavior lands milestone by milestone — no stub logic here.

pub mod ask;
pub mod graph;
pub mod learn;
pub mod search;
pub mod similar;

use crate::cli::Command;

/// Dispatch a parsed command to its handler.
pub fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Learn { text } => learn::run(&text),
        Command::Similar { term } => similar::run(&term),
        Command::Graph { entity } => graph::run(&entity),
        Command::Search { query } => search::run(&query),
        Command::Ask { question } => ask::run(&question),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant dispatches to a handler that succeeds. Handlers print
    /// their status to stdout; cargo test captures that output by default,
    /// so no extra capture machinery is needed.
    #[test]
    fn all_commands_dispatch_successfully() {
        let commands = [
            Command::Learn {
                text: "Rust uses Tokio.".into(),
            },
            Command::Similar {
                term: "rust".into(),
            },
            Command::Graph {
                entity: "rust".into(),
            },
            Command::Search {
                query: "async".into(),
            },
            Command::Ask {
                question: "What does Rust use?".into(),
            },
        ];
        for command in commands {
            assert!(run(command).is_ok());
        }
    }
}
