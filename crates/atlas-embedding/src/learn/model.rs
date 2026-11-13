//! The trained Skip-Gram model: matrix state, token-level read API,
//! incremental growth, and persistence.

use crate::learn::trainer::TrainerConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
    vocab_hash: [u8; 32],
    tokenizer_hash: [u8; 32],
}

#[derive(Serialize, Deserialize)]
struct SkipGramFile {
    version: u32,
    dim: u32,
    vocab_len: u32,
    w_in: Vec<f32>,
    w_out: Vec<f32>,
    vocab_hash: [u8; 32],
    tokenizer_hash: [u8; 32],
}

const FORMAT_VERSION: u32 = 2;

impl SkipGramModel {
    /// An uninitialized model with `dim`-sized rows and no vocabulary rows
    /// yet; `grow_to` fills it. Training-only constructor.
    pub(crate) fn empty(dim: usize) -> Self {
        SkipGramModel {
            dim,
            w_in: Vec::new(),
            w_out: Vec::new(),
            vocab_hash: [0; 32],
            tokenizer_hash: [0; 32],
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
        // Legacy mappings cannot be verified. Preserve the file and require
        // an explicit new destination for a rebuilt model.
        match std::fs::File::open(path) {
            Ok(mut previous) => {
                use std::io::Read;
                let mut version = [0; 4];
                previous.read_exact(&mut version)?;
                if u32::from_le_bytes(version) != FORMAT_VERSION {
                    return Err(LearnerError::Format(
                        "refusing to replace a legacy model; save the rebuilt model to a new path"
                            .into(),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let file = SkipGramFile {
            version: FORMAT_VERSION,
            dim: self.dim as u32,
            vocab_len: self.vocab_len() as u32,
            w_in: self.w_in.clone(),
            w_out: self.w_out.clone(),
            vocab_hash: self.vocab_hash,
            tokenizer_hash: self.tokenizer_hash,
        };
        let bytes = bincode::serialize(&file).expect("SkipGramFile is always serializable");
        atlas_vocab::persistence::write_atomic(path, &bytes)?;
        Ok(())
    }

    /// Load format v2 and verify the ordered vocabulary fingerprint.
    /// Legacy files remain untouched and require retraining to a new path.
    pub fn load(path: &Path, vocab: &atlas_vocab::Vocab) -> Result<Self, LearnerError> {
        let bytes = std::fs::read(path)?;
        if bytes.get(..4) != Some(FORMAT_VERSION.to_le_bytes().as_slice()) {
            return Err(LearnerError::Format("legacy or invalid model format; keep the original file and retrain to a new path (vocabulary mapping cannot be verified)".into()));
        }
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
        if file.vocab_hash != vocab_fingerprint(vocab) {
            return Err(LearnerError::Format(
                "ordered vocabulary does not match the model".into(),
            ));
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
        if file.w_in.iter().chain(&file.w_out).any(|x| !x.is_finite()) {
            return Err(LearnerError::Format(
                "model contains non-finite weights".into(),
            ));
        }
        Ok(SkipGramModel {
            dim,
            w_in: file.w_in,
            w_out: file.w_out,
            vocab_hash: file.vocab_hash,
            tokenizer_hash: file.tokenizer_hash,
        })
    }

    pub(crate) fn bind(
        &mut self,
        vocab: &atlas_vocab::Vocab,
        tokenizer: &atlas_tokenizer::BpeTokenizer,
    ) {
        self.vocab_hash = vocab_fingerprint(vocab);
        self.tokenizer_hash = tokenizer_fingerprint(tokenizer);
    }

    pub(crate) fn check_binding(
        &self,
        vocab: &atlas_vocab::Vocab,
        tokenizer: &atlas_tokenizer::BpeTokenizer,
    ) -> Result<(), LearnerError> {
        if self.vocab_len() != vocab.len() {
            return Err(LearnerError::Format(format!(
                "model has {} rows but vocabulary has {} entries",
                self.vocab_len(),
                vocab.len()
            )));
        }
        if self.vocab_hash != vocab_fingerprint(vocab)
            || self.tokenizer_hash != tokenizer_fingerprint(tokenizer)
        {
            return Err(LearnerError::Format(
                "model vocabulary or tokenizer fingerprint does not match".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn inference_id(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"atlas-skipgram-mean-v1");
        hash.update((self.dim as u64).to_le_bytes());
        hash.update(self.vocab_hash);
        hash.update(self.tokenizer_hash);
        for weight in &self.w_in {
            hash.update(weight.to_le_bytes());
        }
        let suffix: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        format!(
            "atlas-skipgram-d{}-v{}-{suffix}",
            self.dim,
            self.vocab_len()
        )
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

fn vocab_fingerprint(vocab: &atlas_vocab::Vocab) -> [u8; 32] {
    let mut hash = Sha256::new();
    for id in 0..vocab.len() as u32 {
        let token = vocab.token(id).unwrap();
        hash.update((token.len() as u64).to_le_bytes());
        hash.update(token);
    }
    hash.finalize().into()
}
fn tokenizer_fingerprint(tokenizer: &atlas_tokenizer::BpeTokenizer) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(atlas_tokenizer::pretokenize::CL100K_PATTERN.as_bytes());
    let mut entries: Vec<_> = tokenizer.rank_entries().collect();
    entries.sort_by_key(|(_, rank)| *rank);
    for (bytes, rank) in entries {
        hash.update(rank.to_le_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    hash.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with(vectors: Vec<Vec<f32>>) -> SkipGramModel {
        let dim = vectors[0].len();
        SkipGramModel {
            dim,
            w_in: vectors.into_iter().flatten().collect(),
            vocab_hash: [0; 32],
            tokenizer_hash: [0; 32],
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

#[cfg(test)]
mod validation_tests {
    use super::*;
    #[test]
    fn non_finite_model_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad");
        let mut vocab = atlas_vocab::Vocab::default();
        vocab.get_or_insert(b"a");
        let file = SkipGramFile {
            version: FORMAT_VERSION,
            dim: 1,
            vocab_len: 1,
            vocab_hash: vocab_fingerprint(&vocab),
            tokenizer_hash: [0; 32],
            w_in: vec![f32::NAN],
            w_out: vec![0.0],
        };
        std::fs::write(&path, bincode::serialize(&file).unwrap()).unwrap();
        assert!(SkipGramModel::load(&path, &vocab).is_err());
    }
}
