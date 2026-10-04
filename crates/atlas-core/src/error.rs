//! Shared error type for Atlas foundations.
//!
//! Small on purpose: variants appear when a foundation actually needs them.
//! The CLI converts these into `anyhow` errors at the boundary.

/// Library-level error for Atlas foundations.
#[derive(Debug, thiserror::Error)]
pub enum AtlasError {
    /// Loading or parsing configuration failed. Boxed because
    /// `figment::Error` is large (200+ bytes) and this type flows through
    /// many small `Result`s.
    #[error("configuration error: {0}")]
    Config(Box<figment::Error>),

    /// A file or directory operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<figment::Error> for AtlasError {
    fn from(e: figment::Error) -> Self {
        AtlasError::Config(Box::new(e))
    }
}

/// Convenience alias for results returned by Atlas foundations.
pub type Result<T> = std::result::Result<T, AtlasError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_errors_convert_from_figment() {
        let err: AtlasError =
            figment::Error::from(figment::error::Kind::MissingField("log_level".into())).into();
        assert!(err.to_string().contains("configuration error"));
    }

    #[test]
    fn io_errors_convert() {
        let err: AtlasError = std::io::Error::other("disk gone").into();
        assert!(err.to_string().contains("disk gone"));
    }
}
