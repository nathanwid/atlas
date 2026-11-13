//! M08 evaluation: vector vs graph vs hybrid on a crafted fixture.
//!
//! Eight one-chunk documents whose entity graph forms a designed chain
//! (rust -uses-> tokio -requires-> cargo -extends-> rust, python
//! -provides-> pyo3 -requires-> rust) plus filler documents. Queries are
//! lexically close to one document but have gold chunks reachable *only*
//! through graph expansion (zero lexical overlap), so a pure vector scan
//! cannot retrieve them at small k.
//!
//! Embeddings come from a deterministic word-overlap embedder (no model
//! download), so the whole evaluation is offline and bit-reproducible.
//! Assertions cover retrieval mechanics and the designed graph recovery;
//! recall numbers are additionally printed as a report.

use atlas_embedding::Embedder;
use atlas_storage::{HybridConfig, SearchMode, SqliteStore, hybrid_search, learn, search};
use atlas_tokenizer::BpeTokenizer;
use std::collections::HashSet;

/// Word-overlap embeddings: deterministic bag-of-words in a fixed
/// 512-dimensional space. Query/doc similarity = lexical overlap.
struct WordOverlapEmbedder;

const DIM: usize = 512;

fn hash_word(word: &str) -> usize {
    // FNV-1a on the lowercase word.
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in word.to_lowercase().bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash % DIM as u64) as usize
}

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

fn docs() -> Vec<String> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/hybrid_docs.txt"
    );
    std::fs::read_to_string(path)
        .expect("hybrid docs fixture")
        .lines()
        .map(str::to_string)
        .filter(|l| !l.trim().is_empty())
        .collect()
}

fn tokenizer() -> BpeTokenizer {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
    );
    let f = std::fs::File::open(path).expect("ranks fixture");
    BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("ranks parse")
}

async fn learned_store() -> (SqliteStore, Vec<String>) {
    let store = SqliteStore::open_mem().await.unwrap();
    let docs = docs();
    let tok = tokenizer();
    let embedder = WordOverlapEmbedder;
    for doc in &docs {
        learn(&store, &tok, &embedder, doc).await.unwrap();
    }
    (store, docs)
}

fn recall_at_k(
    results: &[atlas_storage::HybridResult],
    gold: &[usize],
    docs: &[String],
    k: usize,
) -> f64 {
    let gold_ids: HashSet<String> = gold
        .iter()
        .map(|&i| format!("{}:0", atlas_storage::doc_id_for(&docs[i])))
        .collect();
    let top: HashSet<String> = results.iter().take(k).map(|r| r.chunk_id.clone()).collect();
    let hits = gold_ids.iter().filter(|g| top.contains(*g)).count();
    hits as f64 / gold.len() as f64
}

fn recall_vector(
    results: &[atlas_storage::SearchResult],
    gold: &[usize],
    docs: &[String],
    k: usize,
) -> f64 {
    let gold_ids: HashSet<String> = gold
        .iter()
        .map(|&i| format!("{}:0", atlas_storage::doc_id_for(&docs[i])))
        .collect();
    let top: HashSet<String> = results.iter().take(k).map(|r| r.chunk_id.clone()).collect();
    let hits = gold_ids.iter().filter(|g| top.contains(*g)).count();
    hits as f64 / gold.len() as f64
}

/// On these queries all vector seeds also have graph evidence. Each seed
/// receives two RRF contributions, so expansion-only hits do not displace
/// them. Fusion can still change their relative order.
#[tokio::test]
async fn shared_support_preserves_fixture_gold() {
    let (store, docs) = learned_store().await;
    for (query, gold) in [
        ("python bindings scripting", [3usize, 0usize]),
        ("tokio async runtimes", [0usize, 1usize]),
    ] {
        let vector = search(&store, &WordOverlapEmbedder, query, 3)
            .await
            .unwrap();
        let hybrid = hybrid_search(
            &store,
            &WordOverlapEmbedder,
            query,
            3,
            &HybridConfig::default(),
            SearchMode::Hybrid,
        )
        .await
        .unwrap();
        let vector_ids: HashSet<&str> = vector.iter().map(|r| r.chunk_id.as_str()).collect();
        let hybrid_ids: HashSet<&str> = hybrid.iter().map(|r| r.chunk_id.as_str()).collect();
        assert_eq!(hybrid_ids.len(), hybrid.len(), "duplicate results");
        assert_eq!(hybrid_ids, vector_ids);
        for hit in &hybrid {
            assert_eq!(hit.source, atlas_storage::ResultSource::Both);
            let seed = vector
                .iter()
                .find(|seed| seed.chunk_id == hit.chunk_id)
                .unwrap();
            assert_eq!(hit.score, Some(seed.score));
            assert!(hit.via.iter().any(|via| via.starts_with("vector rank ")));
            assert!(hit.via.iter().any(|via| via.starts_with("graph rank ")));
        }
        let r_vec = recall_vector(&vector, &gold, &docs, 3);
        let r_hyb = recall_at_k(&hybrid, &gold, &docs, 3);
        println!("{query:?}: vector recall@3={r_vec}, hybrid recall@3={r_hyb}");
        assert_eq!(r_vec, 1.0);
        assert_eq!(r_hyb, 1.0);
    }
}

/// Graph mode returns only expansion chunks; vector mode matches M07
/// search exactly; both modes are deterministic.
#[tokio::test]
async fn modes_match_their_contracts() {
    let (store, docs) = learned_store().await;
    let embedder = WordOverlapEmbedder;
    let config = HybridConfig::default();
    let query = "python bindings";

    let vector = search(&store, &embedder, query, 3).await.unwrap();
    let as_hybrid = hybrid_search(&store, &embedder, query, 3, &config, SearchMode::Vector)
        .await
        .unwrap();
    assert_eq!(vector.len(), as_hybrid.len());
    for (v, h) in vector.iter().zip(as_hybrid.iter()) {
        assert_eq!(v.chunk_id, h.chunk_id);
        assert_eq!(Some(v.score), h.score);
        assert_eq!(h.source, atlas_storage::ResultSource::Vector);
    }

    // k = 3, not 5: with insertion-order rowids the zero-score tie band
    // fills the vector top-5 with every evidence chunk of this fixture
    // (d0, d1, d2 included), leaving graph-only mode without candidates.
    // At k = 3 the cargo and metadata docs stay outside the seeds and the
    // expansion has something to surface.
    let graph = hybrid_search(&store, &embedder, query, 3, &config, SearchMode::Graph)
        .await
        .unwrap();
    assert!(
        !graph.is_empty(),
        "the designed graph must expand something"
    );
    assert!(
        graph
            .iter()
            .all(|r| r.source == atlas_storage::ResultSource::Graph),
        "graph mode returns only expansion chunks"
    );
    assert!(graph.iter().all(|r| r.score.is_none()));

    // Persistence integration: same result after reopen from durable store.
    let dir = tempfile::TempDir::new().unwrap();
    let durable_path = dir.path().join("atlas.db");
    {
        let durable = SqliteStore::open_durable(&durable_path).await.unwrap();
        let tok = tokenizer();
        for doc in &docs {
            learn(&durable, &tok, &embedder, doc).await.unwrap();
        }
    }
    let reopened = SqliteStore::open_durable(&durable_path).await.unwrap();
    let after_restart = hybrid_search(&reopened, &embedder, query, 3, &config, SearchMode::Hybrid)
        .await
        .unwrap();
    let mem_results = hybrid_search(&store, &embedder, query, 3, &config, SearchMode::Hybrid)
        .await
        .unwrap();
    let mem_ids: Vec<&str> = mem_results.iter().map(|r| r.chunk_id.as_str()).collect();
    let disk_ids: Vec<&str> = after_restart.iter().map(|r| r.chunk_id.as_str()).collect();
    assert_eq!(
        mem_ids, disk_ids,
        "restart must not change hybrid retrieval"
    );
    let _ = docs;
}
