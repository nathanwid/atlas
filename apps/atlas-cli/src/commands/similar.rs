//! `atlas similar` — rank entity labels by vector similarity.

use atlas_core::Config;
use atlas_embedding::{Embedder, FastembedEmbedder};
use atlas_storage::{Backend, BruteForceIndex, SqliteStore, VectorIndex};

pub async fn run(config: &Config, term: &str, top: usize) -> anyhow::Result<()> {
    tracing::info!(term, top, "similar invoked");
    anyhow::ensure!(!term.trim().is_empty(), "similar term must not be empty");
    if top == 0 {
        println!("similar: no results requested (top is 0).");
        return Ok(());
    }

    let store = SqliteStore::open_durable(config.data_dir.join("atlas.db")).await?;
    let graph = store.load_graph().await?;
    if graph.entity_count() == 0 {
        println!("similar: no entities learned yet — run `atlas learn <text>` first.");
        return Ok(());
    }

    let exact_match = graph.find_entity(term);
    let candidates: Vec<(u32, String)> = graph
        .entities()
        .filter(|(_, entity)| !entity.display.trim().is_empty())
        .map(|(id, entity)| (id, entity.display.clone()))
        .collect();
    if !candidates.iter().any(|(id, _)| Some(*id) != exact_match) {
        println!(
            "similar: no other entities to compare with {}.",
            quote(term)
        );
        return Ok(());
    }

    let embedder = FastembedEmbedder::bge_small_en_v1_5()?;
    let hits = rank_entities(&candidates, exact_match, term, top, &embedder)?;
    if hits.is_empty() {
        println!("similar: no matching entities.");
        return Ok(());
    }

    println!("similar: labels closest to {}", quote(term));
    for (rank, (label, score)) in hits.iter().enumerate() {
        println!(
            "#{rank} {} cosine={score:.4}",
            super::search::escape_terminal_controls(label)
        );
    }
    Ok(())
}

fn rank_entities(
    candidates: &[(u32, String)],
    exact_match: Option<u32>,
    query: &str,
    top: usize,
    embedder: &dyn Embedder,
) -> anyhow::Result<Vec<(String, f32)>> {
    anyhow::ensure!(!query.trim().is_empty(), "similar term must not be empty");
    if top == 0 {
        return Ok(Vec::new());
    }

    let eligible: Vec<&(u32, String)> = candidates
        .iter()
        .filter(|(id, label)| Some(*id) != exact_match && !label.trim().is_empty())
        .collect();
    if eligible.is_empty() {
        return Ok(Vec::new());
    }

    let labels: Vec<&str> = eligible.iter().map(|(_, label)| label.as_str()).collect();
    let query_vector = embedder.embed_query(query)?;
    let vectors = embedder.embed_documents(&labels)?;
    anyhow::ensure!(
        vectors.len() == labels.len(),
        "embedder returned {} vectors for {} entity labels",
        vectors.len(),
        labels.len()
    );
    let dim = vectors[0].len();
    anyhow::ensure!(dim > 0, "entity embedding dimension must be positive");
    for (index, vector) in vectors.iter().enumerate() {
        anyhow::ensure!(
            vector.len() == dim,
            "entity embedding {index} has dimension {}, expected {dim}",
            vector.len()
        );
        anyhow::ensure!(
            vector.iter().all(|value| value.is_finite()),
            "entity embedding {index} must contain only finite values"
        );
    }
    anyhow::ensure!(
        query_vector.len() == dim && query_vector.iter().all(|value| value.is_finite()),
        "query embedding dimension must be {dim} and all components must be finite"
    );

    let index = BruteForceIndex::build(dim, &vectors)?;
    let hits = index.search(&query_vector, top.min(eligible.len()));
    Ok(hits
        .into_iter()
        .map(|hit| (eligible[hit.id as usize].1.clone(), hit.score))
        .collect())
}

fn quote(text: &str) -> String {
    format!(
        "\"{}\"",
        super::search::escape_terminal_controls(text.trim())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeEmbedder {
        query: Vec<f32>,
        query_error: Option<&'static str>,
        document_error: Option<&'static str>,
        document_count: Option<usize>,
        query_calls: AtomicUsize,
        document_batches: Mutex<Vec<Vec<String>>>,
    }

    impl FakeEmbedder {
        fn new(query: Vec<f32>) -> Self {
            Self {
                query,
                query_error: None,
                document_error: None,
                document_count: None,
                query_calls: AtomicUsize::new(0),
                document_batches: Mutex::new(Vec::new()),
            }
        }

        fn with_query_error(mut self, message: &'static str) -> Self {
            self.query_error = Some(message);
            self
        }

        fn with_document_error(mut self, message: &'static str) -> Self {
            self.document_error = Some(message);
            self
        }

        fn with_document_count(mut self, count: usize) -> Self {
            self.document_count = Some(count);
            self
        }
    }

    impl Embedder for FakeEmbedder {
        fn model_id(&self) -> &str {
            "similar-test-model"
        }

        fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            self.document_batches
                .lock()
                .unwrap()
                .push(texts.iter().map(|text| (*text).to_string()).collect());
            if let Some(message) = self.document_error {
                anyhow::bail!("{message}");
            }
            let mut vectors: Vec<Vec<f32>> = texts
                .iter()
                .map(|text| match *text {
                    "Tokio" | "Cargo" => vec![0.0, 1.0],
                    "Near" => vec![0.8, 0.6],
                    "Bad" => vec![f32::NAN, 0.0],
                    "Wrong" => vec![1.0, 0.0, 0.0],
                    _ => vec![1.0, 0.0],
                })
                .collect();
            if let Some(count) = self.document_count {
                vectors.resize_with(count, || vec![1.0, 0.0]);
            }
            Ok(vectors)
        }

        fn embed_query(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
            self.query_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(message) = self.query_error {
                anyhow::bail!("{message}");
            }
            Ok(self.query.clone())
        }
    }

    #[test]
    fn ranks_by_cosine_excludes_exact_match_and_keeps_stable_ties() {
        let candidates = vec![
            (0, "Rust".to_string()),
            (1, "Tokio".to_string()),
            (2, "Cargo".to_string()),
            (3, "Near".to_string()),
        ];
        let embedder = FakeEmbedder::new(vec![1.0, 0.0]);

        let hits = rank_entities(&candidates, Some(0), " rUsT ", 3, &embedder).unwrap();

        assert_eq!(
            hits.iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec!["Near", "Tokio", "Cargo"]
        );
        assert!((hits[0].1 - 0.8).abs() < 1e-6);
        assert_eq!(
            embedder.document_batches.lock().unwrap().as_slice(),
            &[vec![
                "Tokio".to_string(),
                "Cargo".to_string(),
                "Near".to_string()
            ]]
        );
        assert_eq!(embedder.query_calls.load(Ordering::SeqCst), 1);

        let top_two = rank_entities(
            &candidates,
            Some(0),
            "rust",
            2,
            &FakeEmbedder::new(vec![1.0, 0.0]),
        )
        .unwrap();
        assert_eq!(
            top_two
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec!["Near", "Tokio"]
        );
    }

    #[test]
    fn empty_candidates_and_zero_top_skip_embedding_and_index_search() {
        let embedder = FakeEmbedder::new(vec![1.0, 0.0]);
        assert!(
            rank_entities(&[], None, "rust", 5, &embedder)
                .unwrap()
                .is_empty()
        );
        assert!(
            rank_entities(&[(0, "Rust".into())], None, "rust", 0, &embedder)
                .unwrap()
                .is_empty()
        );
        assert!(
            rank_entities(&[(0, "Rust".into())], Some(0), "rust", 5, &embedder)
                .unwrap()
                .is_empty()
        );
        assert_eq!(embedder.query_calls.load(Ordering::SeqCst), 0);
        assert!(embedder.document_batches.lock().unwrap().is_empty());
    }

    #[test]
    fn rejects_non_finite_and_mismatched_vectors_before_search() {
        let embedder = FakeEmbedder::new(vec![1.0, 0.0]);
        let bad = rank_entities(&[(0, "Bad".into())], None, "rust", 1, &embedder);
        assert!(bad.unwrap_err().to_string().contains("finite"));

        let mismatch = rank_entities(
            &[(0, "Tokio".into()), (1, "Wrong".into())],
            None,
            "rust",
            1,
            &embedder,
        );
        assert!(mismatch.unwrap_err().to_string().contains("dimension"));

        let bad_query = FakeEmbedder::new(vec![f32::NAN, 0.0]);
        let invalid = rank_entities(&[(0, "Tokio".into())], None, "rust", 1, &bad_query);
        assert!(invalid.unwrap_err().to_string().contains("finite"));

        let short_query = FakeEmbedder::new(vec![1.0]);
        let invalid = rank_entities(&[(0, "Tokio".into())], None, "rust", 1, &short_query);
        assert!(invalid.unwrap_err().to_string().contains("dimension"));
    }

    #[test]
    fn query_embedding_failure_keeps_the_cli_error_message() {
        let embedder = FakeEmbedder::new(vec![1.0, 0.0])
            .with_query_error("query embedding failed: model offline");

        let error = rank_entities(&[(0, "Tokio".into())], None, "rust", 5, &embedder).unwrap_err();

        assert_eq!(error.to_string(), "query embedding failed: model offline");
        assert_eq!(embedder.query_calls.load(Ordering::SeqCst), 1);
        assert!(embedder.document_batches.lock().unwrap().is_empty());
    }

    #[test]
    fn label_embedding_failure_keeps_the_cli_error_message() {
        let embedder = FakeEmbedder::new(vec![1.0, 0.0])
            .with_document_error("entity label embedding failed: model unavailable");

        let error = rank_entities(&[(0, "Tokio".into())], None, "rust", 5, &embedder).unwrap_err();

        assert_eq!(
            error.to_string(),
            "entity label embedding failed: model unavailable"
        );
        assert_eq!(embedder.query_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            embedder.document_batches.lock().unwrap().as_slice(),
            &[vec!["Tokio".to_string()]]
        );
    }

    #[test]
    fn incorrect_label_vector_count_returns_a_clear_cli_error() {
        let candidates = vec![(0, "Tokio".into()), (1, "Cargo".into())];
        let embedder = FakeEmbedder::new(vec![1.0, 0.0]).with_document_count(1);

        let error = rank_entities(&candidates, None, "rust", 5, &embedder).unwrap_err();

        assert_eq!(
            error.to_string(),
            "embedder returned 1 vectors for 2 entity labels"
        );
    }
}
