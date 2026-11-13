//! CLI definition for Atlas.
//!
//! The command set mirrors the project vision (`docs/PROJECT.md`): learn,
//! similar, graph, search, ask. Each command's functionality arrives with its
//! own milestone; M01 only wires parsing, logging, and dispatch, so the
//! structure here is the part later milestones extend — subcommands gain
//! flags and handlers gain behavior without redesign.

use clap::{Parser, Subcommand, ValueEnum};

/// Retrieval mode for `atlas search`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SearchModeArg {
    Vector,
    Hybrid,
    Graph,
}

/// Atlas — a semantic learning engine that learns from text.
#[derive(Parser, Debug)]
#[command(name = "atlas", version, about)]
pub struct Cli {
    /// Increase log verbosity: -v = debug, -vv = trace. Overridden by RUST_LOG.
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

/// Atlas commands. Variants correspond 1:1 with modules in `commands/`.
#[derive(Subcommand, Debug, PartialEq)]
pub enum Command {
    /// Learn from a piece of text.
    Learn {
        /// Text to learn from.
        text: String,
    },
    /// Show entities with labels similar to a term.
    Similar {
        /// The term to compare against.
        term: String,
        /// Number of entities to show.
        #[arg(short = 'k', long, default_value_t = 5)]
        top: usize,
    },
    /// Show relationships around an entity.
    Graph {
        /// The entity to explore.
        entity: String,
        /// Maximum graph distance from the entity (0 to 3).
        #[arg(long, default_value_t = 1)]
        hops: usize,
        /// Maximum number of relationship rows to show.
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Search learned content.
    Search {
        /// Free-text query.
        query: String,
        /// Number of results to return.
        #[arg(short = 'k', long, default_value_t = 5)]
        top: usize,
        /// Retrieval mode: hybrid (vector + graph expansion, default),
        /// vector (M07 baseline), or graph (expansion only).
        #[arg(long, value_enum, default_value_t = SearchModeArg::Hybrid)]
        mode: SearchModeArg,
    },
    /// Ask a question over Atlas's own knowledge.
    Ask {
        /// The question to answer.
        question: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("atlas").chain(args.iter().copied()))
    }

    #[test]
    fn parses_learn_with_text() {
        let cli = parse(&["learn", "Rust uses Tokio."]).unwrap();
        assert_eq!(
            cli.command,
            Command::Learn {
                text: "Rust uses Tokio.".to_string()
            }
        );
    }

    #[test]
    fn parses_all_vision_commands() {
        assert!(matches!(
            parse(&["similar", "rust"]).unwrap().command,
            Command::Similar { .. }
        ));
        assert!(matches!(
            parse(&["graph", "rust"]).unwrap().command,
            Command::Graph { .. }
        ));
        let search = parse(&["search", "async"]).unwrap().command;
        let Command::Search { query, top, mode } = search else {
            panic!("expected search");
        };
        assert_eq!(query, "async");
        assert_eq!(top, 5, "default k");
        assert_eq!(mode, SearchModeArg::Hybrid, "default mode is hybrid");
        let search = parse(&["search", "async", "-k", "3", "--mode", "vector"])
            .unwrap()
            .command;
        let Command::Search { top, mode, .. } = search else {
            panic!("expected search");
        };
        assert_eq!(top, 3);
        assert_eq!(mode, SearchModeArg::Vector);
        assert!(matches!(
            parse(&["ask", "What does Rust use?"]).unwrap().command,
            Command::Ask { .. }
        ));
    }

    #[test]
    fn parses_graph_depth_and_limit() {
        let graph = parse(&["graph", "rust"]).unwrap().command;
        let Command::Graph {
            entity,
            hops,
            limit,
        } = graph
        else {
            panic!("expected graph");
        };
        assert_eq!(entity, "rust");
        assert_eq!(hops, 1);
        assert_eq!(limit, 50);

        let graph = parse(&["graph", "rust", "--hops", "2", "--limit", "7"])
            .unwrap()
            .command;
        let Command::Graph { hops, limit, .. } = graph else {
            panic!("expected graph");
        };
        assert_eq!(hops, 2);
        assert_eq!(limit, 7);
    }

    #[test]
    fn parses_similar_top_count() {
        let similar = parse(&["similar", "rust"]).unwrap().command;
        let Command::Similar { term, top } = similar else {
            panic!("expected similar");
        };
        assert_eq!(term, "rust");
        assert_eq!(top, 5);

        let similar = parse(&["similar", "rust", "-k", "3"]).unwrap().command;
        let Command::Similar { top, .. } = similar else {
            panic!("expected similar");
        };
        assert_eq!(top, 3);
    }

    #[test]
    fn verbose_counts() {
        assert_eq!(parse(&["learn", "x"]).unwrap().verbose, 0);
        assert_eq!(parse(&["-v", "learn", "x"]).unwrap().verbose, 1);
        assert_eq!(parse(&["-vv", "learn", "x"]).unwrap().verbose, 2);
        // Global flag: works after the subcommand too.
        assert_eq!(parse(&["learn", "-v", "x"]).unwrap().verbose, 1);
    }

    #[test]
    fn rejects_unknown_command_and_missing_args() {
        assert!(parse(&["fly"]).is_err());
        assert!(parse(&["learn"]).is_err());
        assert!(parse(&[]).is_err());
    }
}
