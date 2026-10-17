//! The QA engine: retrieval, context assembly, and extractive generation.
//!
//! Three clearly separated steps:
//!
//! 1. **Retrieve** — M08 hybrid search (k = 5, hybrid mode).
//! 2. **Assemble context** — split each hit into sentences and score them
//!    by question-word overlap (case-insensitive; deterministic tie-breaks:
//!    overlap desc, vector score desc, chunk id asc).
//! 3. **Generate** — quote the top sentences verbatim with citations, add a
//!    graph lead-in when question words match relationship entities, and
//!    set confidence/insufficiency from the evidence.
//!
//! Thresholds are module constants on purpose: they are pinned by tests
//! with a deterministic embedder rather than tuned at runtime.

use crate::answer::{Answer, AnswerSentence, Confidence};
use atlas_embedding::Embedder;
use atlas_graph::normalize;
use atlas_storage::Store;

/// Hybrid retrieval depth.
const RETRIEVAL_K: usize = 5;
/// Sentences in an answer.
const MAX_SENTENCES: usize = 3;
/// Graph lead-ins per answer.
const MAX_LEAD_INS: usize = 2;

/// Answer `question` from the store's learned content.
pub async fn ask(store: &Store, embedder: &dyn Embedder, question: &str) -> anyhow::Result<Answer> {
    let hits = atlas_storage::hybrid_search(
        store,
        embedder,
        question,
        RETRIEVAL_K,
        &Default::default(),
        atlas_storage::SearchMode::Hybrid,
    )
    .await?;

    let question_words: Vec<String> = question.split_whitespace().map(normalize).collect();
    let mut scored: Vec<(usize, Option<f32>, AnswerSentence)> = Vec::new();
    for hit in &hits {
        for sentence in atlas_graph::extract::sentences(&hit.text) {
            let overlap = sentence
                .split_whitespace()
                .map(normalize)
                .filter(|w| question_words.contains(w))
                .count();
            if overlap == 0 {
                continue; // not on-topic; quoting it would be noise
            }
            scored.push((
                overlap,
                hit.score,
                AnswerSentence {
                    text: sentence.clone(),
                    doc_id: hit.doc_id.clone(),
                    chunk_id: hit.chunk_id.clone(),
                    score: hit.score,
                },
            ));
        }
    }
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.unwrap_or(0.0).total_cmp(&a.1.unwrap_or(0.0)))
            .then_with(|| a.2.chunk_id.cmp(&b.2.chunk_id))
    });
    scored.truncate(MAX_SENTENCES);

    let best_overlap = scored.first().map(|(overlap, _, _)| *overlap).unwrap_or(0);
    let best_score = hits.first().and_then(|h| h.score).unwrap_or(0.0);
    let insufficient = scored.is_empty();

    let confidence = if insufficient {
        Confidence::Low
    } else if best_overlap >= 2 && best_score >= 0.5 {
        Confidence::High
    } else if best_overlap >= 1 {
        Confidence::Medium
    } else {
        Confidence::Low
    };

    let multi_source = {
        let docs: std::collections::HashSet<&str> =
            scored.iter().map(|(_, _, s)| s.doc_id.as_str()).collect();
        docs.len() > 1
    };

    let lead_in = graph_lead_in(store, question).await?;

    Ok(Answer {
        question: question.to_string(),
        lead_in,
        sentences: scored.into_iter().map(|(_, _, s)| s).collect(),
        confidence,
        insufficient,
        multi_source,
    })
}

/// Relationship-shaped questions get a lead-in from the knowledge graph:
/// every relationship whose source or destination entity appears in the
/// question produces one line, in graph insertion order (deterministic),
/// capped at [`MAX_LEAD_INS`].
async fn graph_lead_in(store: &Store, question: &str) -> anyhow::Result<Option<String>> {
    let graph = store.load_graph().await?;
    if graph.relationship_count() == 0 {
        return Ok(None);
    }
    let question_key = normalize(question);
    let mut lines = Vec::new();
    for (_, rel) in graph.relationships() {
        let src = graph.entity_key(rel.src).unwrap_or_default();
        let dst = graph.entity_key(rel.dst).unwrap_or_default();
        if question_key.contains(&src) || question_key.contains(&dst) {
            lines.push(format!(
                "{} -[{}]-> {}.",
                graph
                    .entity(rel.src)
                    .map(|e| e.display.as_str())
                    .unwrap_or("?"),
                graph.predicate(rel.predicate).unwrap_or("?"),
                graph
                    .entity(rel.dst)
                    .map(|e| e.display.as_str())
                    .unwrap_or("?"),
            ));
        }
        if lines.len() >= MAX_LEAD_INS {
            break;
        }
    }
    Ok(if lines.is_empty() {
        None
    } else {
        Some(format!("From the knowledge graph: {}", lines.join(" ")))
    })
}
