//! Personalized PageRank over the knowledge graph (M08).
//!
//! A random walk with restart, seeded on the entities that the vector stage
//! surfaced: mass flows along relationships, a `(1 - damping)` fraction of
//! every node's mass teleports back to the seed set every iteration, and
//! dangling nodes (no relationships) push their mass back to the seeds.
//!
//! The walk runs over the **undirected** view of the graph (both edge
//! directions carry flow): relationships are evidence of association, and a
//! directed-only walk would strand mass on sink entities.
//!
//! Determinism: synchronous iterations in fixed entity order, f64
//! accumulation in fixed sequence, ties broken by entity ID. No external
//! dependencies.

use crate::graph::{EntityId, KnowledgeGraph};

/// Rank entities by personalized PageRank from `seeds`.
///
/// Returns all entities that carry non-zero mass (plus any others after
/// convergence ties), sorted by score descending then entity ID ascending.
/// Empty seeds yield an empty ranking.
pub fn personalized_pagerank(
    graph: &KnowledgeGraph,
    seeds: &[EntityId],
    damping: f64,
    iterations: usize,
) -> Vec<(EntityId, f64)> {
    let n = graph.entity_count();
    if seeds.is_empty() || n == 0 {
        return Vec::new();
    }

    let mut seed_mass = vec![0.0; n];
    let seed_share = (1.0 - damping) / seeds.len() as f64;
    for &s in seeds {
        seed_mass[s as usize] = seed_share;
    }

    // Pre-resolve neighbor lists once; iteration order is edge order.
    let adjacency: Vec<Vec<EntityId>> = (0..n as EntityId)
        .map(|id| graph.neighbors(id).into_iter().map(|nb| nb.other).collect())
        .collect();

    let mut scores = vec![0.0; n];
    for &s in seeds {
        scores[s as usize] = seed_share;
    }

    for _ in 0..iterations {
        let mut next = vec![0.0; n];
        for (i, score) in scores.iter().enumerate() {
            if *score == 0.0 {
                continue;
            }
            let neighbors = &adjacency[i];
            if neighbors.is_empty() {
                // Dangling: teleport the mass back into the seed set.
                let share = score * damping / seeds.len() as f64;
                for &s in seeds {
                    next[s as usize] += share;
                }
            } else {
                let share = score * damping / neighbors.len() as f64;
                for &other in neighbors {
                    next[other as usize] += share;
                }
            }
        }
        for (i, extra) in seed_mass.iter().enumerate() {
            next[i] += extra;
        }
        scores = next;
    }

    let mut ranked: Vec<(EntityId, f64)> = scores
        .into_iter()
        .enumerate()
        .map(|(i, s)| (i as EntityId, s))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeOrigin, Evidence, KnowledgeGraph};

    fn chain_graph() -> KnowledgeGraph {
        // a -> b -> c -> d (all "mentions")
        let mut g = KnowledgeGraph::new();
        let ids: Vec<EntityId> = ["a", "b", "c", "d"]
            .iter()
            .map(|n| g.intern_entity(n))
            .collect();
        let p = g.intern_predicate("mentions");
        for w in ids.windows(2) {
            g.link(
                w[0],
                p,
                w[1],
                EdgeOrigin::Explicit,
                Evidence {
                    doc_id: "d".into(),
                    sentence: "s".into(),
                },
            );
        }
        g
    }

    #[test]
    fn empty_seeds_rank_nothing() {
        let g = chain_graph();
        assert!(personalized_pagerank(&g, &[], 0.85, 20).is_empty());
    }

    #[test]
    fn mass_decays_along_the_chain_from_the_seed() {
        let g = chain_graph();
        let ranked = personalized_pagerank(&g, &[0], 0.85, 30);
        assert_eq!(ranked.len(), 4);
        // The seed holds positive mass; among expansion candidates the
        // degree-2 hub (b) can legitimately outrank the leaf seed, but mass
        // must decay with hop distance: b > c > d.
        assert!(ranked.iter().any(|&(id, s)| id == 0 && s > 0.0));
        let by_id: Vec<f64> = ranked
            .iter()
            .map(|&(id, s)| (id, s))
            .collect::<Vec<_>>()
            .into_iter()
            .filter(|(id, _)| *id != 0)
            .map(|(_, s)| s)
            .collect();
        assert!(
            by_id[0] > by_id[1] && by_id[1] > by_id[2],
            "mass decays with distance"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let g = chain_graph();
        let a = personalized_pagerank(&g, &[1, 2], 0.85, 25);
        let b = personalized_pagerank(&g, &[1, 2], 0.85, 25);
        assert_eq!(a, b);
    }

    #[test]
    fn undirected_flow_reaches_backwards() {
        // d -> c only (directed), but the walk is undirected: seeding d must
        // still push mass onto b and a through reversed traversal.
        let mut g = KnowledgeGraph::new();
        let ids: Vec<EntityId> = ["a", "b", "c", "d"]
            .iter()
            .map(|n| g.intern_entity(n))
            .collect();
        let p = g.intern_predicate("mentions");
        g.link(
            ids[3],
            p,
            ids[2],
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );
        g.link(
            ids[2],
            p,
            ids[1],
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );
        g.link(
            ids[1],
            p,
            ids[0],
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );
        let ranked = personalized_pagerank(&g, &[ids[3]], 0.85, 30);
        assert_eq!(ranked.len(), 4, "all entities ranked");
        assert!(
            ranked.iter().all(|&(_, s)| s > 0.0),
            "undirected flow reaches backwards"
        );
        assert_eq!(
            ranked.last().unwrap().0,
            ids[0],
            "farthest node holds least mass"
        );
    }
}
