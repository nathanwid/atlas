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
use atlas_storage::{HybridConfig, SearchMode, Store, hybrid_search, learn, search};
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

async fn learned_store() -> (Store, Vec<String>) {
    let store = Store::open_mem().await.unwrap();
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

/// Query 1 targets the Python cluster; the second gold chunk (doc 0, the
/// Rust/Tokio doc) has zero lexical overlap and is reachable only through
/// pyo3 -requires-> rust.
#[tokio::test]
async fn hybrid_recovers_graph_only_chunks() {
    let (store, docs) = learned_store().await;
    let embedder = WordOverlapEmbedder;
    let config = HybridConfig::default();

    let query = "python bindings scripting";
    let gold = [3usize, 0usize]; // doc 3 (vector), doc 0 (graph-only)

    let vector = search(&store, &embedder, query, 3).await.unwrap();
    let hybrid = hybrid_search(&store, &embedder, query, 3, &config, SearchMode::Hybrid)
        .await
        .unwrap();

    let dbg_graph = store.load_graph().await.unwrap();
    println!(
        "debug: graph rels={} entities={}",
        dbg_graph.relationship_count(),
        dbg_graph.entity_count()
    );
    for (id, ent) in dbg_graph.entities() {
        println!("debug entity {id} = {:?}", ent.display);
    }
    for (id, rel) in dbg_graph.relationships() {
        println!(
            "debug rel {id}: {} -[{}]-> {} origin={:?} evidence={:?}",
            dbg_graph
                .entity(rel.src)
                .map(|e| e.display.clone())
                .unwrap_or_default(),
            dbg_graph.predicate(rel.predicate).unwrap_or_default(),
            dbg_graph
                .entity(rel.dst)
                .map(|e| e.display.clone())
                .unwrap_or_default(),
            rel.origin,
            rel.evidence
        );
    }
    for s in &vector {
        println!("debug seed {} = {}", s.chunk_id, s.text);
    }

    let r_vec = recall_vector(&vector, &gold, &docs, 3);
    let r_hyb = recall_at_k(&hybrid, &gold, &docs, 3);
    println!("q1 recall@3: vector={r_vec:.3} hybrid={r_hyb:.3}");
    println!(
        "q1 hybrid order: {:?}",
        hybrid
            .iter()
            .map(|r| (r.source, r.chunk_id.clone()))
            .collect::<Vec<_>>()
    );

    // Mechanics: no duplicates; non-graph results are a rank-preserving
    // subset of vector-only search (graph hits may displace some within k —
    // that is the RRF trade-off, pinned by the tokio test below).
    let ids: HashSet<&str> = hybrid.iter().map(|r| r.chunk_id.as_str()).collect();
    assert_eq!(ids.len(), hybrid.len(), "duplicate results");
    let vector_ids: Vec<String> = vector.iter().map(|r| r.chunk_id.clone()).collect();
    let mut last = 0usize;
    for r in hybrid
        .iter()
        .filter(|r| r.source != atlas_storage::ResultSource::Graph)
    {
        let pos = vector_ids[last..]
            .iter()
            .position(|id| id == &r.chunk_id)
            .expect("non-graph result must exist in vector-only search");
        assert!(pos + last >= last);
        last += pos + 1;
    }

    let doc0 = format!("{}:0", atlas_storage::doc_id_for(&docs[0]));
    assert!(
        !vector.iter().take(3).any(|r| r.chunk_id == doc0),
        "designed fixture: doc 0 must be outside the vector top-3"
    );
    assert!(
        hybrid
            .iter()
            .take(3)
            .any(|r| r.chunk_id == doc0 && r.source == atlas_storage::ResultSource::Graph),
        "hybrid must recover the graph-only gold chunk"
    );
    assert!(
        r_hyb > r_vec,
        "designed fixture: hybrid beats vector on this query"
    );
}

/// Query 2 documents the honest RRF trade-off: vector-only is already
/// perfect (1.0), and the graph tier's rank-0 chunk ties the vector rank-0
/// chunk in fused score, so the tie-break can displace a correct vector
/// hit. Hybrid is not claimed to be better — this test pins that it can
/// also be worse, deterministically, on this fixture.
#[tokio::test]
async fn hybrid_can_displace_correct_vector_hits() {
    let (store, docs) = learned_store().await;
    let embedder = WordOverlapEmbedder;
    let config = HybridConfig::default();

    let query = "tokio async runtimes";
    let gold = [0usize, 1usize];

    let vector = search(&store, &embedder, query, 3).await.unwrap();
    let hybrid = hybrid_search(&store, &embedder, query, 3, &config, SearchMode::Hybrid)
        .await
        .unwrap();

    let r_vec = recall_vector(&vector, &gold, &docs, 3);
    let r_hyb = recall_at_k(&hybrid, &gold, &docs, 3);
    println!("q2 recall@3: vector={r_vec:.3} hybrid={r_hyb:.3}");
    println!(
        "q2 hybrid order: {:?}",
        hybrid
            .iter()
            .map(|r| (r.source, r.chunk_id.clone()))
            .collect::<Vec<_>>()
    );

    // The graph stage still worked: the cargo doc (graph-only, zero lexical
    // overlap with the query) entered the results via cargo -extends-> rust.
    let doc2 = format!("{}:0", atlas_storage::doc_id_for(&docs[2]));
    assert!(
        hybrid
            .iter()
            .any(|r| r.chunk_id == doc2 && r.source == atlas_storage::ResultSource::Graph),
        "graph expansion must surface the cargo doc"
    );
    // The trade-off, pinned deterministically for this fixture: the fused
    // ranking loses one of the two gold chunks that vector-only had.
    assert!(
        (r_hyb - 0.5).abs() < 1e-9,
        "deterministic trade-off: hybrid recall@3 = 0.5"
    );
    assert!(
        r_vec > r_hyb,
        "vector-only wins on this already-saturated query"
    );
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

    let graph = hybrid_search(&store, &embedder, query, 5, &config, SearchMode::Graph)
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
    let durable_path = dir.path().join("surreal");
    {
        let durable = Store::open_durable(&durable_path).await.unwrap();
        let tok = tokenizer();
        for doc in &docs {
            learn(&durable, &tok, &embedder, doc).await.unwrap();
        }
    }
    let reopened = Store::open_durable(&durable_path).await.unwrap();
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
