//! Configuration foundation.
//!
//! Layered, later sources win:
//!
//! 1. Built-in defaults (`log_level = "info"`)
//! 2. Optional `atlas.toml` in the current directory
//! 3. Environment variables prefixed `ATLAS_` (e.g. `ATLAS_LOG_LEVEL`)
//!
//! A missing config file is normal (defaults apply); a malformed one is an
//! error. Deliberately minimal: more fields are added only when a milestone
//! actually needs them.

use crate::error::Result;
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Atlas configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Maximum log level (`trace`, `debug`, `info`, `warn`, `error`).
    pub log_level: String,
    /// Directory for durable storage (M07). The SurrealDB database lives at
    /// `<data_dir>/surreal`.
    pub data_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            log_level: "info".to_string(),
            data_dir: PathBuf::from("atlas-data"),
        }
    }
}

impl Config {
    /// Load from the standard sources: defaults, `atlas.toml` (cwd), env.
    pub fn load() -> Result<Self> {
        Self::from_figment(
            Figment::new()
                .join(Serialized::defaults(Config::default()))
                .merge(Toml::file("atlas.toml"))
                .merge(Env::prefixed("ATLAS_")),
        )
    }

    /// Load from an explicit file path; `None` skips the file layer.
    /// Split from [`Config::load`] so tests can compose sources without
    /// touching the working directory or environment.
    pub fn load_from(file: Option<&Path>) -> Result<Self> {
        let mut figment = Figment::new().join(Serialized::defaults(Config::default()));
        if let Some(file) = file {
            figment = figment.merge(Toml::file(file));
        }
        Self::from_figment(figment.merge(Env::prefixed("ATLAS_")))
    }

    fn from_figment(figment: Figment) -> Result<Self> {
        Ok(figment.extract()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn defaults_apply_without_file() {
        let config = Config::load_from(None).expect("defaults load");
        assert_eq!(config, Config::default());
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn file_overrides_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atlas.toml");
        fs::write(&path, "log_level = \"debug\"\n").unwrap();
        let config = Config::load_from(Some(&path)).expect("file loads");
        assert_eq!(config.log_level, "debug");
    }

    #[test]
    fn missing_field_in_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atlas.toml");
        fs::write(&path, "unrelated_key = true\n").unwrap();
        // "unrelated_key" is unknown, but log_level comes from defaults, so
        // extraction still succeeds — unknown keys are ignored by design.
        let config = Config::load_from(Some(&path)).expect("unknown keys ignored");
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn malformed_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atlas.toml");
        fs::write(&path, "log_level = ").unwrap();
        let result = Config::load_from(Some(&path));
        assert!(matches!(result, Err(crate::AtlasError::Config(_))));
    }

    #[test]
    fn invalid_level_value_type_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atlas.toml");
        fs::write(&path, "log_level = 3\n").unwrap();
        let result = Config::load_from(Some(&path));
        assert!(matches!(result, Err(crate::AtlasError::Config(_))));
    }
}
