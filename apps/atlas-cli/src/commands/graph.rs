//! `atlas graph` — explore stored relationships around an entity.

use atlas_core::Config;
use atlas_storage::{Backend, SqliteStore};
use std::collections::{HashMap, HashSet};

const MAX_HOPS: usize = 3;

pub async fn run(config: &Config, entity: &str, hops: usize, limit: usize) -> anyhow::Result<()> {
    tracing::info!(entity, hops, limit, "graph invoked");
    validate_options(entity, hops, limit)?;
    let store = SqliteStore::open_durable(config.data_dir.join("atlas.db")).await?;
    println!(
        "{}",
        render_neighborhood(&store, entity, hops, limit).await?
    );
    Ok(())
}

async fn render_neighborhood(
    store: &impl Backend,
    entity: &str,
    hops: usize,
    limit: usize,
) -> anyhow::Result<String> {
    validate_options(entity, hops, limit)?;

    let graph = store.load_graph().await?;
    let Some(center) = graph.find_entity(entity) else {
        if graph.entity_count() == 0 {
            return Ok("graph: no entities learned yet — run `atlas learn <text>` first.".into());
        }
        return Ok(format!(
            "graph: entity '{}' was not found — learn text that contains it first.",
            super::search::escape_terminal_controls(entity.trim())
        ));
    };

    let reachable = graph.reachable(center, hops);
    let mut in_neighborhood = HashSet::with_capacity(reachable.len() + 1);
    in_neighborhood.insert(center);
    in_neighborhood.extend(reachable.iter().copied());

    // Visit entities in deterministic breadth-first order. Neighbor IDs are
    // ordered by relationship insertion, and `seen_relationships` only tests
    // membership; it never controls output order.
    let mut entity_order = Vec::with_capacity(reachable.len() + 1);
    entity_order.push(center);
    entity_order.extend(reachable);
    let mut seen_relationships = HashSet::new();
    let mut relationship_ids = Vec::new();
    for id in entity_order {
        for neighbor in graph.neighbors(id) {
            if in_neighborhood.contains(&neighbor.other)
                && seen_relationships.insert(neighbor.relationship)
            {
                relationship_ids.push(neighbor.relationship);
            }
        }
    }

    let center_name = graph
        .entity(center)
        .map(|item| item.display.as_str())
        .unwrap_or("?");
    let hop_unit = if hops == 1 { "hop" } else { "hops" };
    let mut output = format!(
        "graph: {} (within {hops} {hop_unit})",
        super::search::escape_terminal_controls(center_name)
    );
    if relationship_ids.is_empty() {
        output.push_str("\n  no relationships in this neighborhood.");
        return Ok(output);
    }

    let mut evidence_chunks = HashMap::new();
    for relationship_id in relationship_ids.iter().take(limit) {
        let Some(relationship) = graph.relationship(*relationship_id) else {
            continue;
        };
        let source = graph
            .entity(relationship.src)
            .map(|item| item.display.as_str())
            .unwrap_or("?");
        let predicate = graph.predicate(relationship.predicate).unwrap_or("?");
        let destination = graph
            .entity(relationship.dst)
            .map(|item| item.display.as_str())
            .unwrap_or("?");
        let direction = if relationship.src == center {
            "outgoing"
        } else if relationship.dst == center {
            "incoming"
        } else {
            "within neighborhood"
        };
        output.push_str(&format!(
            "\n  {} -[{}]-> {} ({}, {})",
            super::search::escape_terminal_controls(source),
            super::search::escape_terminal_controls(predicate),
            super::search::escape_terminal_controls(destination),
            relationship.origin,
            direction,
        ));

        let mut live_evidence = None;
        for evidence in &relationship.evidence {
            if evidence.sentence.trim().is_empty() {
                continue;
            }
            if !evidence_chunks.contains_key(&evidence.doc_id) {
                evidence_chunks.insert(
                    evidence.doc_id.clone(),
                    store.get_chunk(&evidence.doc_id).await?,
                );
            }
            let Some(Some(chunk)) = evidence_chunks.get(&evidence.doc_id) else {
                continue;
            };
            if chunk.text.contains(&evidence.sentence) {
                live_evidence = Some((
                    chunk.doc_id.clone(),
                    evidence.doc_id.clone(),
                    evidence.sentence.clone(),
                ));
                break;
            }
        }

        if let Some((doc_id, chunk_id, sentence)) = live_evidence {
            output.push_str(&format!(
                "\n    evidence: \"{}\" [doc {}, chunk {}]",
                super::search::escape_terminal_controls(&sentence),
                super::search::escape_terminal_controls(&doc_id),
                super::search::escape_terminal_controls(&chunk_id),
            ));
        } else {
            output.push_str("\n    evidence unavailable");
        }
    }
    if relationship_ids.len() > limit {
        output.push_str(&format!(
            "\n  {} additional relationship(s) not shown.",
            relationship_ids.len() - limit
        ));
    }
    Ok(output)
}

fn validate_options(entity: &str, hops: usize, limit: usize) -> anyhow::Result<()> {
    anyhow::ensure!(!entity.trim().is_empty(), "graph entity must not be empty");
    anyhow::ensure!(
        hops <= MAX_HOPS,
        "graph hops must be between 0 and {MAX_HOPS}"
    );
    anyhow::ensure!(limit > 0, "graph limit must be greater than zero");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_embedding::Embedder;
    use atlas_storage::{Backend, ChunkRecord, SqliteStore};
    use atlas_tokenizer::BpeTokenizer;

    struct FakeEmbedder;

    impl Embedder for FakeEmbedder {
        fn model_id(&self) -> &str {
            "graph-test-model"
        }

        fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| vec![text.len() as f32, 1.0])
                .collect())
        }
    }

    async fn learned_store(texts: &[&str]) -> SqliteStore {
        let store = SqliteStore::open_mem().await.unwrap();
        let tokenizer = BpeTokenizer::embedded_cl100k();
        for text in texts {
            atlas_storage::learn(&store, &tokenizer, &FakeEmbedder, text)
                .await
                .unwrap();
        }
        store
    }

    #[tokio::test]
    async fn neighborhood_is_bounded_ordered_and_includes_both_directions() {
        let store = learned_store(&[
            "Rust uses Tokio.",
            "Cargo depends on Rust.",
            "Tokio uses Rust.",
            "Tokio supports Futures.",
            "Python mentions Cargo.",
        ])
        .await;

        let one_hop = render_neighborhood(&store, "  rUsT  ", 1, 50)
            .await
            .unwrap();
        let outgoing = one_hop.find("Rust -[uses]-> Tokio").unwrap();
        let incoming = one_hop.find("Cargo -[depends on]-> Rust").unwrap();
        assert!(outgoing < incoming, "relationship order must be stable");
        assert!(one_hop.contains("outgoing"));
        assert!(one_hop.contains("incoming"));
        assert!(one_hop.contains("evidence: \"Rust uses Tokio"));
        assert!(one_hop.contains("[doc "));
        assert!(!one_hop.contains("Tokio -[supports]-> Futures"));
        assert_eq!(
            one_hop,
            render_neighborhood(&store, "Rust", 1, 50).await.unwrap()
        );

        let two_hops = render_neighborhood(&store, "Rust", 2, 50).await.unwrap();
        assert!(two_hops.contains("Tokio -[supports]-> Futures"));
        assert!(two_hops.contains("Python -[mentions]-> Cargo"));

        let limited = render_neighborhood(&store, "Rust", 2, 1).await.unwrap();
        assert_eq!(edge_rows(&limited).len(), 1);
        assert!(limited.contains("additional relationship"));
    }

    #[tokio::test]
    async fn stale_evidence_keeps_the_edge_but_hides_text_and_citation() {
        let store = learned_store(&["Rust uses Tokio."]).await;
        let graph = store.load_graph().await.unwrap();
        let evidence_id = graph.relationships().next().unwrap().1.evidence[0]
            .doc_id
            .clone();
        let old_chunk = store.get_chunk(&evidence_id).await.unwrap().unwrap();
        store
            .upsert_chunk(&ChunkRecord {
                chunk_id: old_chunk.chunk_id,
                doc_id: old_chunk.doc_id,
                ordinal: old_chunk.ordinal,
                text: "The source text has changed.".into(),
            })
            .await
            .unwrap();

        let rendered = render_neighborhood(&store, "Rust", 1, 50).await.unwrap();
        assert!(rendered.contains("Rust -[uses]-> Tokio"));
        assert!(rendered.contains("evidence unavailable"));
        assert!(!rendered.contains("Rust uses Tokio."));
        assert!(!rendered.contains("[doc "));

        store.delete_chunk(&evidence_id).await.unwrap();
        let rendered_after_delete = render_neighborhood(&store, "Rust", 1, 50).await.unwrap();
        assert!(rendered_after_delete.contains("Rust -[uses]-> Tokio"));
        assert!(rendered_after_delete.contains("evidence unavailable"));
        assert!(!rendered_after_delete.contains("Rust uses Tokio."));
        assert!(!rendered_after_delete.contains("[doc "));
    }

    #[tokio::test]
    async fn unknown_entity_returns_a_clear_message() {
        let store = learned_store(&["Rust uses Tokio."]).await;
        let rendered = render_neighborhood(&store, "Python", 1, 50).await.unwrap();
        assert!(rendered.contains("entity 'Python' was not found"));
    }

    #[test]
    fn rejects_empty_names_and_unbounded_options() {
        assert!(validate_options("  ", 1, 50).is_err());
        assert!(validate_options("Rust", MAX_HOPS + 1, 50).is_err());
        assert!(validate_options("Rust", 1, 0).is_err());
    }

    fn edge_rows(rendered: &str) -> Vec<&str> {
        rendered
            .lines()
            .filter(|line| line.starts_with("  ") && line.contains(" -["))
            .collect()
    }
}
