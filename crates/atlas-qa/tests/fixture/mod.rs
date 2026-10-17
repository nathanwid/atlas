//! M09 QA evaluation fixture.
//!
//! Six one-chunk documents with a designed relationship chain (same shape
//! as the M08 fixture) plus a deterministic word-overlap embedder, so every
//! answer, citation, and confidence level is reproducible offline.

use atlas_embedding::Embedder;
use atlas_qa::{Answer, ask};
use atlas_storage::{Store, learn};
use atlas_tokenizer::BpeTokenizer;

pub const DIM: usize = 512;

pub fn hash_word(word: &str) -> usize {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in word.to_lowercase().bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash % DIM as u64) as usize
}

pub struct WordOverlapEmbedder;

impl Embedder for WordOverlapEmbedder {
    fn model_id(&self) -> &str {
        "word-overlap"
    }

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0; DIM];
                for word in text.split_whitespace() {
                    v[hash_word(word)] += 1.0;
                }
                v
            })
            .collect())
    }
}

pub fn docs() -> Vec<String> {
    concat!(
        "Rust uses Tokio for async runtimes.\n",
        "Tokio requires Cargo metadata files.\n",
        "Cargo extends Rust tooling.\n",
        "Python provides PyO3 bindings.\n",
        "Rust requires PyO3 for bindings.\n",
        "Atlas is a learning engine project.\n",
    )
    .lines()
    .map(str::to_string)
    .collect()
}

pub fn tokenizer() -> BpeTokenizer {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
    );
    let f = std::fs::File::open(path).expect("ranks fixture");
    BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("ranks parse")
}

pub async fn learned_store() -> Store {
    let store = Store::open_mem().await.unwrap();
    let tok = tokenizer();
    let embedder = WordOverlapEmbedder;
    for doc in docs() {
        learn(&store, &tok, &embedder, &doc).await.unwrap();
    }
    store
}

/// Ask and require a non-insufficient answer with at least one cited
/// sentence from `expected_doc` (by document index).
pub async fn ask_expect_doc(store: &Store, question: &str, expected_doc: usize) -> Answer {
    let answer = ask(store, &WordOverlapEmbedder, question).await.unwrap();
    assert!(
        !answer.insufficient,
        "question {question:?} must find evidence"
    );
    let doc_id = atlas_storage::doc_id_for(&docs()[expected_doc]);
    assert!(
        answer.sentences.iter().any(|s| s.doc_id == doc_id),
        "question {question:?} must cite doc {expected_doc}"
    );
    answer
}
