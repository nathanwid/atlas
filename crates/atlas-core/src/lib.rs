//! Shared foundations for Atlas: configuration and error types.
//!
//! Kept minimal; modules appear here when a milestone actually needs them.

pub mod config;
pub mod error;

pub use config::Config;
pub use error::{AtlasError, Result};
