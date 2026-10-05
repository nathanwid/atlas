//! Corpus smoke test: extraction over the shared corpus fixture must run,
//! produce a non-trivial graph, and be fully deterministic across runs.
//! Quality is report-only (printed probes).

use atlas_graph::{EdgeOrigin, Extractor, ExtractorConfig, KnowledgeGraph};

fn corpus() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/corpus.txt"
    ))
    .expect("corpus fixture")
}

fn extract_all() -> KnowledgeGraph {
    let extractor = Extractor::new(ExtractorConfig::default());
    let mut graph = KnowledgeGraph::new();
    // One document, same as a real ingestion pass would do (doc_id is the
    // M08 hook for chunk-level evidence).
    extractor.extract("corpus", &corpus(), &mut graph);
    graph
}

#[test]
fn corpus_extraction_is_deterministic_and_nontrivial() {
    let a = extract_all();
    let b = extract_all();

    assert!(
        a.entity_count() >= 3,
        "corpus mentions several capitalized terms"
    );
    // Zero relationships is a legitimate outcome for this corpus: its
    // sentences rarely contain two recognized entities. The smoke test
    // asserts determinism and structural validity, not relationship volume.
    assert_eq!(a.entity_count(), b.entity_count());
    let mut pairs_a: Vec<_> = a
        .relationships()
        .map(|(rid, r)| (rid, r.src, r.predicate, r.dst, r.origin, r.evidence.clone()))
        .collect();
    let mut pairs_b: Vec<_> = b
        .relationships()
        .map(|(rid, r)| (rid, r.src, r.predicate, r.dst, r.origin, r.evidence.clone()))
        .collect();
    pairs_a.sort();
    pairs_b.sort();
    assert_eq!(pairs_a, pairs_b, "extraction must be deterministic");
}

#[test]
fn corpus_graph_report() {
    let graph = extract_all();
    println!(
        "corpus graph: {} entities, {} relationships ({} explicit, {} co-occurrence)",
        graph.entity_count(),
        graph.relationship_count(),
        graph
            .relationships()
            .filter(|(_, r)| r.origin == EdgeOrigin::Explicit)
            .count(),
        graph
            .relationships()
            .filter(|(_, r)| r.origin == EdgeOrigin::CoOccurrence)
            .count()
    );
    for (id, entity) in graph.entities() {
        let degree = graph.neighbors(id).len();
        if degree > 0 {
            println!("  {:?} (degree {})", entity.display, degree);
        }
    }
    // Mechanical sanity only: every relationship endpoint exists.
    for (_, rel) in graph.relationships() {
        assert!((rel.src as usize) < graph.entity_count());
        assert!((rel.dst as usize) < graph.entity_count());
    }
}
