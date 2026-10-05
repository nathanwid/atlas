//! The learned model as a text embedder.
//!
//! [`LearnedEmbedder`] implements [`Embedder`] on top of a
//! [`SkipGramModel`]: tokenize → vocabulary IDs → mean of the known
//! tokens' learned vectors. Mean pooling *is* the sentence-embedding
//! semantics for now — simple, deterministic, and honest about what it is.
//! Texts with no known tokens embed to the zero vector.
//!
//! Cache note: `model_id` encodes dim and vocabulary size, not training
//! state. Retraining with the same vocabulary length under the same
//! `model_id` can reuse cached vectors from an older model — a documented
//! limitation, not a correctness failure (caching learned embeddings is
//! opt-in, unlike the FastEmbed path).

use crate::embedder::Embedder;
use crate::learn::model::SkipGramModel;
use atlas_tokenizer::BpeTokenizer;
use atlas_tokenizer::pretokenize::pretokenize;
use atlas_vocab::Vocab;

/// Text-level inference over a learned Skip-Gram model.
#[derive(Debug, Clone)]
pub struct LearnedEmbedder {
    model: SkipGramModel,
    tokenizer: BpeTokenizer,
    vocab: Vocab,
    model_id: String,
}

impl LearnedEmbedder {
    /// Pair a trained model with the tokenizer and vocabulary it was
    /// trained against.
    pub fn new(model: SkipGramModel, tokenizer: BpeTokenizer, vocab: Vocab) -> Self {
        if model.vocab_len() != vocab.len() {
            panic!(
                "model has {} rows but vocabulary has {} entries",
                model.vocab_len(),
                vocab.len()
            );
        }
        let model_id = format!("atlas-skipgram-d{}-v{}", model.dim(), vocab.len());
        LearnedEmbedder {
            model,
            tokenizer,
            vocab,
            model_id,
        }
    }

    /// The trained model, e.g. to persist after further `learn` calls.
    pub fn model(&self) -> &SkipGramModel {
        &self.model
    }

    fn text_vector(&self, text: &str) -> Vec<f32> {
        let dim = self.model.dim();
        let mut sum = vec![0f32; dim];
        let mut known = 0usize;
        for piece in pretokenize(text) {
            for rank in self.tokenizer.encode(piece) {
                let Some(bytes) = self.tokenizer.token_bytes(rank) else {
                    continue;
                };
                let Some(id) = self.vocab.get(bytes) else {
                    continue; // unknown to this model: skipped, not an error
                };
                let Some(v) = self.model.token_vector(id) else {
                    continue;
                };
                for (s, x) in sum.iter_mut().zip(v) {
                    *s += x;
                }
                known += 1;
            }
        }
        if known > 0 {
            for s in &mut sum {
                *s /= known as f32;
            }
        }
        sum
    }
}

impl Embedder for LearnedEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.text_vector(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::trainer::{TrainerConfig, train};

    fn tokenizer() -> BpeTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
        );
        let f = std::fs::File::open(path).expect("ranks fixture");
        BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("parses")
    }

    fn trained() -> (LearnedEmbedder, Vocab) {
        let tok = tokenizer();
        let mut vocab = Vocab::default();
        let config = TrainerConfig {
            dim: 8,
            window: 2,
            negatives: 3,
            lr: 0.1,
            epochs: 5,
            seed: 3,
        };
        let model = train(&mut vocab, &tok, "alpha beta alpha gamma", &config);
        let embedder = LearnedEmbedder::new(model, tok, vocab.clone());
        (embedder, vocab)
    }

    #[test]
    fn known_text_is_mean_of_token_vectors() {
        let (embedder, vocab) = trained();
        let tok = tokenizer();

        let ids: Vec<u32> = ["alpha", " beta"]
            .iter()
            .flat_map(|p| tok.encode(p))
            .map(|r| vocab.get(tok.token_bytes(r).unwrap()).unwrap())
            .collect();
        let manual: Vec<f32> = (0..8)
            .map(|j| {
                ids.iter()
                    .map(|&id| embedder.model().token_vector(id).unwrap()[j])
                    .sum::<f32>()
                    / ids.len() as f32
            })
            .collect();

        assert_eq!(
            embedder.embed_documents(&["alpha beta"]).unwrap()[0],
            manual
        );
    }

    #[test]
    fn unknown_tokens_are_skipped_not_errors() {
        let (embedder, _) = trained();
        // The training corpus only knows "alpha", " beta", " gamma"; any
        // other token's bytes are absent from the vocabulary.
        let mixed = embedder
            .embed_documents(&["alpha delta"])
            .unwrap()
            .remove(0);
        let alpha_only = embedder.embed_documents(&["alpha"]).unwrap().remove(0);
        assert_eq!(mixed, alpha_only, "unknown token must not change the mean");
    }

    #[test]
    fn fully_unknown_text_is_zero_vector() {
        let (embedder, _) = trained();
        let vec = embedder.embed_documents(&["delta"]).unwrap().remove(0);
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn model_id_encodes_dim_and_vocab() {
        let (embedder, vocab) = trained();
        assert_eq!(
            embedder.model_id(),
            format!("atlas-skipgram-d8-v{}", vocab.len())
        );
    }

    #[test]
    #[should_panic(expected = "rows but vocabulary")]
    fn mismatched_vocab_is_rejected() {
        let (embedder, _) = trained();
        let empty_vocab = Vocab::default();
        // Rebuild with an empty vocab: row count mismatch must be loud.
        let _ = LearnedEmbedder::new(embedder.model().clone(), tokenizer(), {
            let mut v = empty_vocab;
            v.get_or_insert(b"unrelated");
            v
        });
    }
}
