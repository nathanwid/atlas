//! Chunk embedding with pluggable backends.
//!
//! [`Embedder`] is the boundary that keeps Atlas honest: the chunk+embed
//! pipeline only knows this trait, so the real model, a counting fake in
//! tests, or a future backend are interchangeable. `FastembedEmbedder` wraps
//! the fastembed crate (ONNX Runtime); it exists so Atlas compares against a
//! real embedding model instead of a toy one.
//!
//! The trait separates documents from queries: several retrieval-tuned
//! models (BGE among them) embed queries with a model-specific prefix while
//! documents stay raw. Backends that don't distinguish get the default
//! `embed_query`, which is plain single-text embedding; model-specific
//! prefixes live inside the backend that needs them.

use std::sync::Mutex;

pub trait Embedder {
    /// Stable identifier of the model, used as part of cache keys so vectors
    /// from different models can never mix.
    fn model_id(&self) -> &str;

    /// Embed a batch of document texts. Implementations may batch
    /// internally; the slice is the unit of work, and output order must
    /// match input order.
    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embed a single search query. The default treats a query like any
    /// text; backends whose model family distinguishes queries from
    /// documents override this.
    fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.embed_documents(&[text])?.remove(0))
    }
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

/// BGE family recipe: retrieval queries carry this instruction prefix;
/// documents do not. Kept here, next to the only backend that needs it.
const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

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

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
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

    fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut model = self
            .model
            .lock()
            .expect("embedder lock poisoned by a panic in embed");
        let mut out = model.embed(&[format!("{BGE_QUERY_PREFIX}{text}")], None)?;
        anyhow::ensure!(!out.is_empty(), "embedder returned no vector for the query");
        Ok(out.remove(0))
    }
}
