//! Personalized PageRank scaling benchmarks (hardening Phase 2).
//!
//! Synthetic deterministic graphs: a ring over N entities plus LCG-chosen
//! chords, giving approximately N edges in a connected, non-degenerate
//! topology. Graph construction happens once per benchmark (setup, untimed);
//! each iteration runs one full personalized_pagerank call (20 iterations,
//! damping 0.85, two seeds).

use atlas_graph::{EntityId, KnowledgeGraph, personalized_pagerank};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

/// Build a connected synthetic graph with approximately `edge_target`
/// directed relationships: a ring over the nodes plus LCG-chosen chords.
fn synthetic_graph(edge_target: usize) -> KnowledgeGraph {
    let nodes = edge_target - edge_target / 10; // ring consumes nodes edges
    let chords = edge_target - nodes;

    let mut graph = KnowledgeGraph::new();
    let names: Vec<String> = (0..nodes).map(|i| format!("entity{i}")).collect();
    for name in &names {
        graph.intern_entity(name);
    }
    let predicate = graph.intern_predicate("mentions");

    // Ring: keeps the graph connected.
    for i in 0..nodes as EntityId {
        let next = (i + 1) % nodes as EntityId;
        graph.link(
            i,
            predicate,
            next,
            atlas_graph::EdgeOrigin::Explicit,
            atlas_graph::Evidence {
                doc_id: "bench".into(),
                sentence: "ring".into(),
            },
        );
    }
    // Chords: deterministic pseudo-random skips (may dedup a little; the
    // edge count is "approximately" the target, as documented).
    let mut state = 0x9E3779B97F4A7C15u64;
    for _ in 0..chords {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let src = (state % nodes as u64) as EntityId;
        let dst = ((state >> 16) % nodes as u64) as EntityId;
        if src != dst {
            graph.link(
                src,
                predicate,
                dst,
                atlas_graph::EdgeOrigin::CoOccurrence,
                atlas_graph::Evidence {
                    doc_id: "bench".into(),
                    sentence: "chord".into(),
                },
            );
        }
    }
    graph
}

fn bench_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("personalized_pagerank");
    for edge_target in [1000usize, 10_000] {
        // Setup (untimed): graph construction.
        let graph = synthetic_graph(edge_target);
        let seeds: Vec<EntityId> = vec![0, graph.entity_count() as EntityId / 2];

        group.throughput(Throughput::Elements(graph.relationship_count() as u64));
        group.bench_function(format!("{edge_target}_edges"), |b| {
            b.iter(|| personalized_pagerank(&graph, &seeds, 0.85, 20))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pagerank);
criterion_main!(benches);
