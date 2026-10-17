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
fn init_logging(config: &Config, verbose: u8) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level = match verbose {
            0 => config.log_level.as_str(),
            1 => "debug",
            _ => "trace",
        };
        EnvFilter::new(level)
    });
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
