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

    /// Validate backend-specific limits, including before cache/duplicate hits.
    fn validate_documents(&self, _texts: &[&str]) -> anyhow::Result<()> {
        Ok(())
    }

    /// Embed a batch of document texts. Implementations may batch
    /// internally; the slice is the unit of work, and output order must
    /// match input order.
    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embed a single search query. The default treats a query like any
    /// text; backends whose model family distinguishes queries from
    /// documents override this.
    fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let vectors = self.embed_documents(&[text])?;
        anyhow::ensure!(
            vectors.len() == 1,
            "embedder must return exactly one query vector"
        );
        Ok(vectors.into_iter().next().unwrap())
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
    max_tokens: usize,
}

/// BGE family recipe: retrieval queries carry this instruction prefix;
/// documents do not. Kept here, next to the only backend that needs it.
const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

impl FastembedEmbedder {
    /// Load `BAAI/bge-small-en-v1.5`. Downloads the ONNX weights (~100 MB)
    /// into fastembed's cache directory on first use.
    pub fn bge_small_en_v1_5() -> anyhow::Result<Self> {
        let mut model = fastembed::TextEmbedding::try_new(
            fastembed::TextInitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(true),
        )?;
        let max_tokens = model
            .tokenizer
            .get_truncation()
            .ok_or_else(|| anyhow::anyhow!("BGE tokenizer has no configured token limit"))?
            .max_length;
        model
            .tokenizer
            .with_truncation(None)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(Self {
            model: Mutex::new(model),
            model_id: "BAAI/bge-small-en-v1.5".into(),
            max_tokens,
        })
    }
}

impl Embedder for FastembedEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn validate_documents(&self, texts: &[&str]) -> anyhow::Result<()> {
        let model = self.model.lock().expect("embedder lock poisoned");
        for (index, text) in texts.iter().enumerate() {
            let count = model
                .tokenizer
                .encode(*text, true)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .len();
            check_budget("document", index, count, self.max_tokens)?;
        }
        Ok(())
    }

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_documents(texts)?;
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
        let prefixed = format!("{BGE_QUERY_PREFIX}{text}");
        let count = model
            .tokenizer
            .encode(prefixed.as_str(), true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .len();
        check_budget("query", 0, count, self.max_tokens)?;
        let mut out = model.embed(&[prefixed], None)?;
        anyhow::ensure!(!out.is_empty(), "embedder returned no vector for the query");
        Ok(out.remove(0))
    }
}

fn check_budget(kind: &str, index: usize, count: usize, limit: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        count <= limit,
        "BGE {kind} {} uses {count} model tokens; limit is {limit}, including special tokens and the query prefix. Use shorter {kind} text; split documents into shorter passages and learn them separately.",
        index + 1
    );
    Ok(())
}
#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn exact_limit_is_valid_and_overflow_is_actionable() {
        assert!(check_budget("document", 0, 512, 512).is_ok());
        let error = check_budget("query", 0, 513, 512).unwrap_err().to_string();
        assert!(error.contains("query") && error.contains("513") && error.contains("shorter"));
    }
}
