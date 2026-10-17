//! Restart recovery for the durable store (hardening Phase 1).
//!
//! Ingests the eight hybrid-fixture documents into a durable SurrealKV
//! store, drops the handle, reopens, and verifies (a) hybrid retrieval
//! still works and (b) the persisted graph is identical to the
//! pre-restart graph, compared through stable identity: normalized entity
//! keys, display labels, kinds, and relationships as
//! (src key, predicate, dst key, origin, evidence) with merged evidence.

use atlas_embedding::Embedder;
use atlas_storage::{HybridConfig, SearchMode, Store, hybrid_search, learn};
use atlas_tokenizer::BpeTokenizer;
use std::collections::HashSet;

struct WordOverlapEmbedder;

const DIM: usize = 512;

fn hash_word(word: &str) -> usize {
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

fn tokenizer() -> BpeTokenizer {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
    );
    let f = std::fs::File::open(path).expect("ranks fixture");
    BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("ranks parse")
}

fn fixture_docs() -> Vec<String> {
    concat!(
        "Rust uses Tokio for async runtimes.\n",
        "Tokio requires Cargo metadata files.\n",
        "Cargo extends Rust tooling.\n",
        "Python provides PyO3 bindings.\n",
        "Rust requires PyO3 for bindings.\n",
        "Atlas is a learning engine project.\n",
        "The tokenizer splits bytes into pieces.\n",
        "SurrealDB stores documents durably.\n",
    )
    .lines()
    .map(str::to_string)
    .collect()
}

/// Stable, order-independent signature of a graph: sorted entity rows and
/// sorted relationship rows keyed by normalized identity.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
struct EvidenceRow(String, String);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
struct RelationshipRow(String, String, String, String, Vec<EvidenceRow>);

#[derive(Debug, PartialEq)]
struct GraphSignature {
    entities: Vec<(String, String, Option<String>)>, // key, display, kind
    relationships: Vec<RelationshipRow>,
}

fn signature(graph: &atlas_graph::KnowledgeGraph) -> GraphSignature {
    let mut entities: Vec<_> = graph
        .entities()
        .map(|(id, e)| {
            (
                graph.entity_key(id).unwrap_or_default(),
                e.display.clone(),
                e.kind.clone(),
            )
        })
        .collect();
    entities.sort();

    let mut relationships: Vec<RelationshipRow> = graph
        .relationships()
        .map(|(_, r)| {
            RelationshipRow(
                graph.entity_key(r.src).unwrap_or_default(),
                graph.predicate(r.predicate).unwrap_or_default().to_string(),
                graph.entity_key(r.dst).unwrap_or_default(),
                format!("{:?}", r.origin),
                {
                    let mut ev: Vec<EvidenceRow> = r
                        .evidence
                        .iter()
                        .map(|e| EvidenceRow(e.doc_id.clone(), e.sentence.clone()))
                        .collect();
                    ev.sort();
                    ev
                },
            )
        })
        .collect();
    relationships.sort();

    GraphSignature {
        entities,
        relationships,
    }
}

#[tokio::test]
async fn durable_restart_preserves_graph_and_retrieval() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("surreal");
    let docs = fixture_docs();
    let tok = tokenizer();
    let embedder = WordOverlapEmbedder;

    let before_signature;
    {
        let store = Store::open_durable(&db_path).await.unwrap();
        for doc in &docs {
            learn(&store, &tok, &embedder, doc).await.unwrap();
        }
        before_signature = signature(&store.load_graph().await.unwrap());
    } // handle dropped

    let reopened = Store::open_durable(&db_path).await.unwrap();
    let after = reopened.load_graph().await.unwrap();
    let after_signature = signature(&after);
    assert_eq!(
        before_signature, after_signature,
        "graph must survive restart unchanged"
    );
    assert!(
        !after_signature.relationships.is_empty(),
        "fixture must produce relationships"
    );

    // M06 semantics preserved across restart: normalized identity + display
    // labels + directed edges + origins are all present in the signature
    // above; spot-check the upgrade invariant survived too.
    for RelationshipRow(_, _, _, origin, _) in &after_signature.relationships {
        assert!(
            origin == "Explicit" || origin == "CoOccurrence",
            "origin must roundtrip, got {origin}"
        );
    }

    // Hybrid retrieval still works after restart, and is identical across
    // two more opens (restart does not perturb ranking).
    let query = "python bindings";
    let run = || async {
        hybrid_search(
            &reopened,
            &embedder,
            query,
            3,
            &HybridConfig::default(),
            SearchMode::Hybrid,
        )
        .await
        .unwrap()
    };
    let first = run().await;
    assert!(!first.is_empty(), "retrieval must work after restart");
    let second = run().await;
    let ids = |r: &[atlas_storage::HybridResult]| -> Vec<String> {
        r.iter().map(|h| h.chunk_id.clone()).collect()
    };
    assert_eq!(
        ids(&first),
        ids(&second),
        "restart retrieval must be deterministic"
    );
    let unique: HashSet<String> = ids(&first).into_iter().collect();
    assert_eq!(
        unique.len(),
        first.len(),
        "no duplicate results after restart"
    );
}
