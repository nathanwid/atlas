//! Atlas CLI entry point: logging, configuration, dispatch.
//!
//! Startup order matters: logging needs the configured level, so the config
//! loads first (silently, before the subscriber exists) and only then is
//! logging initialized. After that, everything is logged and dispatched.

mod cli;
mod commands;

use atlas_core::Config;
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    let config = Config::load()?;
    init_logging(&config, args.verbose);
    tracing::debug!(command = ?args.command, "dispatching");
    commands::run(&config, args.command).await
}

/// Initialize `tracing` from RUST_LOG, falling back to -v/-vv, falling back
/// to the configured `log_level`.
///
/// Unless RUST_LOG is set explicitly, the SurrealDB SDK's own log targets
/// are silenced: its shutdown path emits ERROR-level task-cancellation
/// noise on every process exit, and its engine INFO lines add no value at
/// CLI level. Explicit RUST_LOG restores full SDK logging.
fn init_logging(config: &Config, verbose: u8) {
    let filter = match EnvFilter::try_from_default_env() {
        Ok(env_filter) => env_filter,
        Err(_) => {
            let level = match verbose {
                0 => config.log_level.as_str(),
                1 => "debug",
                _ => "trace",
            };
            EnvFilter::new(format!(
                "{level},surrealdb=off,surrealdb_core=off,surrealkv=off"
            ))
        }
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
