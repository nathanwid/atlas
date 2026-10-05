//! Atlas's own learned embeddings (M05 Learning Engine).
//!
//! Skip-Gram with negative sampling, hand-rolled: no autodiff, no tensor
//! library, single-threaded, deterministic. It exists so Atlas owns an
//! embedding model it fully understands; the pretrained FastEmbed path
//! (see [`crate::embedder`]) remains the external baseline.
//!
//! Layering:
//!
//! * [`trainer::TrainerConfig`] + [`train`] — the training API. Consumes
//!   tokenized text as `TokenId` streams from M03's vocabulary — one
//!   token-ID system, no second registry. Vocabulary growth during
//!   training goes through `Vocab::get_or_insert`.
//! * [`SkipGramModel`] — trainable model state (`W_in`/`W_out`, row-major
//!   `V × d`) plus the token-level read API: `token_vector`, `similarity`,
//!   `nearest`.
//! * [`LearnedEmbedder`] — the inference adapter that implements
//!   [`crate::Embedder`] for text: tokenize → known token vectors → mean
//!   pooling. Mean pooling is the entire sentence-embedding story for now;
//!   anything smarter is future work.
//!
//! Determinism: fixed iteration order, linear lr decay, and a seeded
//! xorshift PRNG (no `rand` dependency) make identical inputs produce
//! bit-identical matrices.

pub mod embedder;
pub mod model;
pub mod trainer;

pub use embedder::LearnedEmbedder;
pub use model::{LearnerError, SkipGramModel};
pub use trainer::{TrainerConfig, train};
