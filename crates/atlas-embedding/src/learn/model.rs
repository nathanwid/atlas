//! The trained Skip-Gram model: matrix state, token-level read API,
//! incremental growth, and persistence.

use crate::learn::trainer::TrainerConfig;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Errors from loading a trained model.
#[derive(Debug, thiserror::Error)]
pub enum LearnerError {
    /// File I/O failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The file exists but is not a usable model, or does not match the
    /// vocabulary it is being loaded against.
    #[error("model file is invalid: {0}")]
    Format(String),
}

/// A trained Skip-Gram model.
///
/// Two matrices, row-major, one row per vocabulary ID: `w_in` holds center
/// ("word") vectors — the ones exposed for similarity — and `w_out` holds
/// context vectors, kept so training can resume. Rows are indexed by
/// `TokenId`, which is why a model is only meaningful against the
/// vocabulary it was trained with; [`SkipGramModel::load`] enforces that.
#[derive(Debug, Clone)]
pub struct SkipGramModel {
    dim: usize,
    w_in: Vec<f32>,
    w_out: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct SkipGramFile {
    version: u32,
    dim: u32,
    vocab_len: u32,
    w_in: Vec<f32>,
    w_out: Vec<f32>,
}

const FORMAT_VERSION: u32 = 1;

impl SkipGramModel {
    /// An uninitialized model with `dim`-sized rows and no vocabulary rows
    /// yet; `grow_to` fills it. Training-only constructor.
    pub(crate) fn empty(dim: usize) -> Self {
        SkipGramModel {
            dim,
            w_in: Vec::new(),
            w_out: Vec::new(),
        }
    }

    /// Row count (== vocabulary length at training time).
    pub fn vocab_len(&self) -> usize {
        self.w_in.len() / self.dim
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The learned (center) vector for a token ID.
    pub fn token_vector(&self, id: u32) -> Option<&[f32]> {
        let start = id as usize * self.dim;
        if id as usize >= self.vocab_len() {
            return None;
        }
        Some(&self.w_in[start..start + self.dim])
    }

    /// Cosine similarity between two tokens' learned vectors. Returns
    /// `None` when either ID is out of range; zero-length vectors score 0.
    pub fn similarity(&self, a: u32, b: u32) -> Option<f32> {
        let va = self.token_vector(a)?;
        let vb = self.token_vector(b)?;
        Some(cosine(va, vb))
    }

    /// The `k` most similar tokens to `id` (excluding itself), best first.
    pub fn nearest(&self, id: u32, k: usize) -> Option<Vec<(u32, f32)>> {
        let v = self.token_vector(id)?;
        let mut hits: Vec<(u32, f32)> = (0..self.vocab_len() as u32)
            .filter(|&other| other != id)
            .map(|other| (other, cosine(v, self.token_vector(other).unwrap())))
            .collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hits.truncate(k);
        Some(hits)
    }

    /// Grow the matrices to cover `vocab.len()` rows; new rows are
    /// initialized exactly like training-time init, seeded from the
    /// configuration and the current row count so the result is
    /// reproducible across sessions. Existing rows are untouched.
    pub(crate) fn grow_to(&mut self, target_len: usize, config: &TrainerConfig) {
        let v = self.vocab_len();
        if target_len <= v {
            return;
        }
        // One fresh stream per growth event: deterministic given (seed, old
        // row count), independent of how many times growth happens.
        let mut rng =
            crate::learn::trainer::XorShift::new(config.seed.wrapping_add(v as u64).max(1));
        for _ in 0..(target_len - v) {
            self.w_in.extend(
                (0..self.dim)
                    .map(|_| rng.next_range(-0.5 / self.dim as f32, 0.5 / self.dim as f32)),
            );
            self.w_out.extend(std::iter::repeat_n(0.0, self.dim));
        }
    }

    /// Write the model to `path` in the versioned bincode format.
    pub fn save(&self, path: &Path) -> Result<(), LearnerError> {
        let file = SkipGramFile {
            version: FORMAT_VERSION,
            dim: self.dim as u32,
            vocab_len: self.vocab_len() as u32,
            w_in: self.w_in.clone(),
            w_out: self.w_out.clone(),
        };
        let bytes = bincode::serialize(&file).expect("SkipGramFile is always serializable");
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a model and check it against `vocab`: row count must equal the
    /// vocabulary length, or the IDs this model was trained on are not the
    /// IDs the caller would use.
    pub fn load(path: &Path, vocab: &atlas_vocab::Vocab) -> Result<Self, LearnerError> {
        let bytes = std::fs::read(path)?;
        let file: SkipGramFile =
            bincode::deserialize(&bytes).map_err(|e| LearnerError::Format(e.to_string()))?;
        if file.version != FORMAT_VERSION {
            return Err(LearnerError::Format(format!(
                "unsupported version {} (expected {})",
                file.version, FORMAT_VERSION
            )));
        }
        if file.vocab_len as usize != vocab.len() {
            return Err(LearnerError::Format(format!(
                "model was trained against {} vocabulary entries, but the provided vocabulary has {}",
                file.vocab_len,
                vocab.len()
            )));
        }
        let dim = file.dim as usize;
        if dim == 0
            || file.w_in.len() != dim * file.vocab_len as usize
            || file.w_out.len() != dim * file.vocab_len as usize
        {
            return Err(LearnerError::Format(
                "matrix sizes do not match dim and vocab_len".to_string(),
            ));
        }
        Ok(SkipGramModel {
            dim,
            w_in: file.w_in,
            w_out: file.w_out,
        })
    }

    /// Access to context vectors for training only.
    pub(crate) fn matrices_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.w_in, &mut self.w_out)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with(vectors: Vec<Vec<f32>>) -> SkipGramModel {
        let dim = vectors[0].len();
        SkipGramModel {
            dim,
            w_in: vectors.into_iter().flatten().collect(),
            w_out: vec![0.0; 0], // read API only touches w_in
        }
    }

    #[test]
    fn token_vector_and_similarity_basics() {
        let m = model_with(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
        assert_eq!(m.token_vector(0), Some(&[1.0, 0.0][..]));
        assert_eq!(m.token_vector(9), None);
        assert!((m.similarity(0, 1).unwrap() - 0.0).abs() < 1e-6);
        assert!((m.similarity(0, 2).unwrap() - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        assert_eq!(m.similarity(0, 42), None);
    }

    #[test]
    fn nearest_excludes_self_and_sorts_desc() {
        let m = model_with(vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.8, 0.2],
        ]);
        let hits = m.nearest(0, 10).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, 1);
        assert_eq!(hits[1].0, 3);
        assert_eq!(hits[2].0, 2);
        assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn zero_vector_scores_zero_not_nan() {
        let m = model_with(vec![vec![1.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(m.similarity(0, 1).unwrap(), 0.0);
    }
}
