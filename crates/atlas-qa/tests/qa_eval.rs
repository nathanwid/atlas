//! M09 QA evaluation: grounding, citations, traceability, insufficiency,
//! confidence buckets, multi-source provenance, determinism, and graph
//! lead-ins — all offline and deterministic.

mod fixture;

use std::collections::HashSet;

use atlas_qa::{Confidence, ask};
use atlas_storage::{Store, doc_id_for};
use fixture::{ask_expect_doc, docs, learned_store};

#[tokio::test]
async fn answers_are_grounded_with_accurate_citations() {
    let store = learned_store().await;
    let answer = ask_expect_doc(&store, "What uses Tokio for async runtimes?", 0).await;

    // Grounding: every sentence is a verbatim substring of a learned chunk.
    for sentence in &answer.sentences {
        let chunk = store
            .get_chunk(&sentence.chunk_id)
            .await
            .unwrap()
            .expect("cited chunk exists");
        assert_eq!(chunk.chunk_id, sentence.chunk_id, "citation accuracy");
        assert_eq!(chunk.doc_id, sentence.doc_id, "document citation accuracy");
        assert!(
            chunk.text.contains(&sentence.text),
            "answer sentence must be a verbatim substring of the cited chunk"
        );
    }
    assert!(!answer.sentences.is_empty());

    // Traceability: question → sentence → chunk → document all connected.
    let doc_id = doc_id_for(&docs()[0]);
    assert!(answer.sentences.iter().any(|s| s.doc_id == doc_id));
}

#[tokio::test]
async fn relationship_questions_get_graph_lead_ins() {
    let store = learned_store().await;
    let answer = ask_expect_doc(&store, "What uses Tokio for async runtimes?", 0).await;
    // "uses" + "tokio" appear in the question: the uses edge is a lead-in.
    assert!(
        answer
            .lead_in
            .as_deref()
            .is_some_and(|l| l.contains("Rust -[uses]-> Tokio")),
        "expected a graph lead-in for the uses relationship, got {:?}",
        answer.lead_in
    );
}

#[tokio::test]
async fn non_relationship_questions_have_no_lead_in() {
    let store = learned_store().await;
    // "learning engine" touches no relationship entity.
    let answer = ask(
        &store,
        &fixture::WordOverlapEmbedder,
        "learning engine project",
    )
    .await
    .unwrap();
    assert!(!answer.insufficient);
    assert!(answer.lead_in.is_none(), "no entity match, no lead-in");
}

#[tokio::test]
async fn insufficient_evidence_is_explicit() {
    let store = learned_store().await;
    // Zero lexical overlap with everything learned.
    let answer = ask(
        &store,
        &fixture::WordOverlapEmbedder,
        "quantum xylophone aerobics",
    )
    .await
    .unwrap();
    assert!(answer.insufficient);
    assert!(answer.sentences.is_empty());
    assert_eq!(answer.confidence, Confidence::Low);

    let empty = Store::open_mem().await.unwrap();
    let answer = ask(&empty, &fixture::WordOverlapEmbedder, "anything at all")
        .await
        .unwrap();
    assert!(answer.insufficient);
}

#[tokio::test]
async fn confidence_buckets_are_deterministic() {
    let store = learned_store().await;
    // Strong overlap question.
    let strong = ask_expect_doc(&store, "What uses Tokio for async runtimes?", 0).await;
    assert_eq!(
        strong.confidence,
        Confidence::High,
        "overlap >= 2 and strong score"
    );
    // Determinism: identical question, identical answer.
    let again = ask(
        &store,
        &fixture::WordOverlapEmbedder,
        "What uses Tokio for async runtimes?",
    )
    .await
    .unwrap();
    assert_eq!(again, strong);
}

#[tokio::test]
async fn multi_source_answers_report_provenance_multiplicity() {
    let store = learned_store().await;
    // "rust" matches docs 0, 2, 4 — evidence from multiple documents.
    let answer = ask(
        &store,
        &fixture::WordOverlapEmbedder,
        "Rust tooling bindings",
    )
    .await
    .unwrap();
    assert!(!answer.insufficient);
    if answer.multi_source {
        let unique_docs: HashSet<&str> =
            answer.sentences.iter().map(|s| s.doc_id.as_str()).collect();
        assert!(
            unique_docs.len() > 1,
            "multi_source implies sentences from multiple docs"
        );
    }
}
