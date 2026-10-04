//! Chunk embedding with pluggable backends.
//!
//! [`Embedder`] is the boundary that keeps Atlas honest: the chunk+embed
//! pipeline only knows this trait, so the real model, a counting fake in
//! tests, or a future backend are interchangeable. `FastembedEmbedder` wraps
//! the fastembed crate (ONNX Runtime); it exists so Atlas compares against a
//! real embedding model instead of a toy one.

use std::sync::Mutex;

pub trait Embedder {
    /// Stable identifier of the model, used as part of cache keys so vectors
    /// from different models can never mix.
    fn model_id(&self) -> &str;

    /// Embed a batch of texts. Implementations may batch internally; the
    /// slice is the unit of work, and output order must match input order.
    fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// Embedder backed by fastembed's ONNX bge-small-en-v1.5 (384 dimensions).
///
/// fastembed's `TextEmbedding::embed` takes `&mut self`, so the model sits
/// behind a mutex: the [`Embedder`] trait is object-safe with `&self`, which
/// is what the pipeline (and later the service layer) wants. Embedding is
/// blocking CPU work, so a std mutex is the right lock, not an async one.
pub struct FastembedEmbedder {
    model: Mutex<fastembed::TextEmbedding>,
    model_id: String,
}

impl FastembedEmbedder {
    /// Load `BAAI/bge-small-en-v1.5`. Downloads the ONNX weights (~100 MB)
    /// into fastembed's cache directory on first use.
    pub fn bge_small_en_v1_5() -> anyhow::Result<Self> {
        Ok(Self {
            model: Mutex::new(fastembed::TextEmbedding::try_new(
                fastembed::TextInitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
                    .with_show_download_progress(true),
            )?),
            model_id: "BAAI/bge-small-en-v1.5".to_string(),
        })
    }
}

impl Embedder for FastembedEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // batch_size None: let fastembed pick its internal batch size.
        let mut model = self
            .model
            .lock()
            .expect("embedder lock poisoned by a panic in embed");
        Ok(model.embed(texts, None)?)
    }
}
