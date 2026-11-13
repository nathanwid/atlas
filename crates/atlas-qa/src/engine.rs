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
use atlas_storage::Backend;
use atlas_storage::SqliteStore;

/// Hybrid retrieval depth.
const RETRIEVAL_K: usize = 5;
/// Sentences in an answer.
const MAX_SENTENCES: usize = 3;
/// Graph lead-ins per answer.
const MAX_LEAD_INS: usize = 2;

/// Answer `question` from the store's learned content.
pub async fn ask(
    store: &SqliteStore,
    embedder: &dyn Embedder,
    question: &str,
) -> anyhow::Result<Answer> {
    let hits = atlas_storage::hybrid_search(
        store,
        embedder,
        question,
        RETRIEVAL_K,
        &Default::default(),
        atlas_storage::SearchMode::Hybrid,
    )
    .await?;

    let question_words = content_words(question);
    let mut scored: Vec<(usize, Option<f32>, AnswerSentence)> = Vec::new();
    for hit in &hits {
        for sentence in atlas_graph::extract::sentences(&hit.text) {
            let overlap = content_words(&sentence)
                .intersection(&question_words)
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
    let best_score = scored
        .first()
        .and_then(|(_, score, _)| *score)
        .unwrap_or(0.0);
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

    let lead_in = graph_lead_in(store, question, &hits).await?;

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
/// matching relationships need at least one live supporting chunk. Select
/// the first valid source in evidence order and cite it. Relationship order
/// is deterministic and the output is capped at [`MAX_LEAD_INS`].
async fn graph_lead_in(
    store: &SqliteStore,
    question: &str,
    hits: &[atlas_storage::HybridResult],
) -> anyhow::Result<Option<String>> {
    let graph = store.load_graph().await?;
    if graph.relationship_count() == 0 {
        return Ok(None);
    }
    let question_key = normalize(question);
    // Reuse retrieved text. Cache additional lookups, including missing
    // chunks, because several relationships can share an evidence source.
    let mut chunks: std::collections::HashMap<String, Option<(String, String)>> = hits
        .iter()
        .map(|hit| {
            (
                hit.chunk_id.clone(),
                Some((hit.doc_id.clone(), hit.text.clone())),
            )
        })
        .collect();
    let mut lines = Vec::new();
    for (_, rel) in graph.relationships() {
        let src = graph.entity_key(rel.src).unwrap_or_default();
        let dst = graph.entity_key(rel.dst).unwrap_or_default();
        if !(mentions_entity(&question_key, &src) || mentions_entity(&question_key, &dst)) {
            continue;
        }
        for ev in &rel.evidence {
            if ev.sentence.trim().is_empty() {
                continue;
            }
            let chunk = match chunks.entry(ev.doc_id.clone()) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let chunk = store.get_chunk(&ev.doc_id).await?;
                    entry.insert(chunk.map(|chunk| (chunk.doc_id, chunk.text)))
                }
            };
            let Some((doc_id, text)) = chunk else {
                continue;
            };
            if !text.contains(&ev.sentence) {
                continue;
            }
            lines.push(format!(
                "{} -[{}]-> {}. [doc {}, chunk {}]",
                graph
                    .entity(rel.src)
                    .map(|e| e.display.as_str())
                    .unwrap_or("?"),
                graph.predicate(rel.predicate).unwrap_or("?"),
                graph
                    .entity(rel.dst)
                    .map(|e| e.display.as_str())
                    .unwrap_or("?"),
                doc_id,
                ev.doc_id,
            ));
            break;
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

/// Inputs use graph normalization. Check the boundaries of the complete
/// entity phrase so, for example, Go does not match Cargo.
fn mentions_entity(question: &str, entity: &str) -> bool {
    !entity.is_empty()
        && question.match_indices(entity).any(|(start, _)| {
            let end = start + entity.len();
            let is_word = |c: char| c.is_alphanumeric() || c == '_';
            question[..start]
                .chars()
                .next_back()
                .is_none_or(|c| !is_word(c))
                && question[end..].chars().next().is_none_or(|c| !is_word(c))
        })
}

/// QA normalization is separate from lossless tokenization and graph identity.
fn content_words(text: &str) -> std::collections::BTreeSet<String> {
    const FUNCTION_WORDS: &[&str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "what", "which", "who", "when",
        "where", "why", "how", "do", "does", "did", "of", "to", "in", "on", "at", "for", "from",
        "with", "by", "and", "or", "it", "this", "that", "has", "have",
    ];
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .filter(|w| !FUNCTION_WORDS.contains(&w.as_str()))
        .collect()
}
