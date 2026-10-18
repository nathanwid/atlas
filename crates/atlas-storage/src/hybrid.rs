//! Hybrid vector + graph retrieval (M08).
//!
//! Two stages, kept as separate tiers rather than blended into one score:
//!
//! 1. **Vector stage** (identical to M07's [`search`]): exact scan over
//!    persisted embeddings → seed chunks with true cosine scores.
//! 2. **Graph stage**: entities evidenced by the seed chunks seed a
//!    personalized PageRank walk over the knowledge graph; expansion
//!    entities' relationship evidence points at candidate chunks, ranked by
//!    incident PPR mass.
//!
//! Tiering is deliberate: vector scores stay real cosine similarities, and
//! graph-only hits follow ranked by PPR mass — nothing is multiplied into a
//! fabricated unified number. A chunk found by both stages is marked
//! [`ResultSource::Both`] but keeps its vector position.
//!
//! Fallback: an empty graph or no seed entities degrades hybrid to plain
//! vector results. Missing chunk records are skipped and counted, never a
//! hard failure — evidence may reference chunks removed elsewhere.
//!
//! Determinism: entity collection follows relationship insertion order,
//! PPR ties break by entity ID, chunk ties break by chunk ID.

use crate::backend::Backend;
use crate::workflow;
use atlas_embedding::Embedder;
use atlas_graph::{KnowledgeGraph, personalized_pagerank};

/// Knobs for the graph stage. Defaults are sensible for Atlas-scale graphs.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// How many relationship hops to expand from seed entities.
    pub expand_hops: usize,
    /// PageRank damping factor (probability of following a relationship
    /// rather than teleporting back to the seeds).
    pub ppr_damping: f64,
    /// Synchronous PageRank iterations.
    pub ppr_iterations: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        HybridConfig {
            expand_hops: 1,
            ppr_damping: 0.85,
            ppr_iterations: 20,
        }
    }
}

/// Which retrieval stage(s) produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultSource {
    Vector,
    Graph,
    Both,
}

impl std::fmt::Display for ResultSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResultSource::Vector => f.write_str("vector"),
            ResultSource::Graph => f.write_str("graph"),
            ResultSource::Both => f.write_str("both"),
        }
    }
}

/// Retrieval mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// Vector stage only — the M07 baseline.
    Vector,
    /// Vector seeds + graph expansion (default).
    #[default]
    Hybrid,
    /// Vector seeds anchor the walk; only graph-expanded chunks are
    /// returned. The honest single-signal graph comparison.
    Graph,
}

/// One hybrid result with its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridResult {
    pub chunk_id: String,
    pub doc_id: String,
    /// Real cosine score for vector-stage hits; `None` for graph-only hits.
    pub score: Option<f32>,
    pub text: String,
    pub source: ResultSource,
    /// Human-readable provenance: the vector score and/or the entity path
    /// that pulled the chunk in.
    pub via: Vec<String>,
}

/// Expand seed entities through the graph, bounded by hops, cycle-safe.
fn expand_entities(graph: &KnowledgeGraph, seeds: &[u32], hops: usize) -> Vec<u32> {
    let mut in_set = vec![false; graph.entity_count()];
    let mut order = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    for &s in seeds {
        if !in_set[s as usize] {
            in_set[s as usize] = true;
            order.push(s);
            queue.push_back((s, 0usize));
        }
    }
    while let Some((current, depth)) = queue.pop_front() {
        if depth == hops {
            continue;
        }
        for neighbor in graph.neighbors(current) {
            let n = neighbor.other as usize;
            if in_set[n] {
                continue;
            }
            in_set[n] = true;
            order.push(neighbor.other);
            queue.push_back((neighbor.other, depth + 1));
        }
    }
    order
}

/// Entities evidenced by the given chunks, in relationship insertion order.
fn entities_for_chunks<'a>(
    graph: &'a KnowledgeGraph,
    chunk_ids: &std::collections::HashSet<&str>,
) -> Vec<(u32, Vec<&'a str>)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (_, rel) in graph.relationships() {
        for ev in &rel.evidence {
            if chunk_ids.contains(ev.doc_id.as_str()) {
                for id in [rel.src, rel.dst] {
                    if seen.insert(id) {
                        out.push((id, Vec::new()));
                    }
                }
            }
        }
    }
    out
}

/// Run retrieval in `mode`.
///
/// Fusion is rank-based (reciprocal rank fusion, standard constant 60):
/// the vector stage ranks seeds by cosine, the graph stage ranks expansion
/// chunks by PPR mass, and a chunk's fused score is the sum of
/// `1 / (60 + rank)` from each stage it appears in. Ranks are real, so no
/// fabricated similarity numbers enter the ordering; every result carries
/// its per-stage provenance in `via`. Top `k` fused results are returned.
///
/// `vector` skips the graph stage entirely (the M07 baseline); `graph`
/// returns only the expansion tier, PPR-ranked. Missing chunk records are
/// skipped during expansion; cycles are handled by the visited set; the
/// result of one mode never depends on map iteration order.
pub async fn hybrid_search(
    store: &impl Backend,
    embedder: &dyn Embedder,
    query: &str,
    k: usize,
    config: &HybridConfig,
    mode: SearchMode,
) -> anyhow::Result<Vec<HybridResult>> {
    const RRF_CONSTANT: f64 = 60.0;

    struct Fused {
        doc_id: String,
        text: Option<String>,
        vector_rank: Option<usize>,
        vector_score: Option<f32>,
        graph_rank: Option<usize>,
        graph_via: Vec<String>,
    }

    let seeds = workflow::search(store, embedder, query, k).await?;
    let mut fused: std::collections::HashMap<String, Fused> = std::collections::HashMap::new();
    for (rank, s) in seeds.iter().enumerate() {
        fused.insert(
            s.chunk_id.clone(),
            Fused {
                doc_id: s.doc_id.clone(),
                text: Some(s.text.clone()),
                vector_rank: Some(rank),
                vector_score: Some(s.score),
                graph_rank: None,
                graph_via: Vec::new(),
            },
        );
    }

    if mode != SearchMode::Vector {
        let graph = store.load_graph().await?;
        let seed_chunk_ids: std::collections::HashSet<&str> =
            seeds.iter().map(|s| s.chunk_id.as_str()).collect();
        let seed_entities: Vec<u32> = entities_for_chunks(&graph, &seed_chunk_ids)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        if graph.relationship_count() > 0 && !seed_entities.is_empty() {
            let expanded = expand_entities(&graph, &seed_entities, config.expand_hops);
            let ranked = personalized_pagerank(
                &graph,
                &seed_entities,
                config.ppr_damping,
                config.ppr_iterations,
            );
            let ppr: std::collections::HashMap<u32, f64> = ranked.into_iter().collect();

            // Candidate chunks: evidence of relationships incident to
            // expanded entities, scored by the best incident entity mass.
            // Seed chunks stay in the vector tier; fusion marks them Both.
            let expanded_set: std::collections::HashSet<u32> = expanded.into_iter().collect();
            let mut best: std::collections::HashMap<String, (f64, Vec<String>)> =
                std::collections::HashMap::new();
            for (_, rel) in graph.relationships() {
                if !(expanded_set.contains(&rel.src) || expanded_set.contains(&rel.dst)) {
                    continue;
                }
                let mass = ppr
                    .get(&rel.src)
                    .copied()
                    .unwrap_or(0.0)
                    .max(ppr.get(&rel.dst).copied().unwrap_or(0.0));
                let path = format!(
                    "{} -[{}]-> {}",
                    graph
                        .entity(rel.src)
                        .map(|e| e.display.as_str())
                        .unwrap_or("?"),
                    graph.predicate(rel.predicate).unwrap_or("?"),
                    graph
                        .entity(rel.dst)
                        .map(|e| e.display.as_str())
                        .unwrap_or("?"),
                );
                for ev in &rel.evidence {
                    if seed_chunk_ids.contains(ev.doc_id.as_str()) {
                        continue;
                    }
                    if store.get_chunk(&ev.doc_id).await?.is_none() {
                        continue; // evidence references a missing chunk; skip
                    }
                    let entry = best.entry(ev.doc_id.clone()).or_insert((0.0, Vec::new()));
                    if mass > entry.0 {
                        entry.0 = mass;
                    }
                    if !entry.1.contains(&path) {
                        entry.1.push(path.clone());
                    }
                }
            }
            let mut graph_tier: Vec<(String, f64, Vec<String>)> = best
                .into_iter()
                .map(|(id, (mass, paths))| (id, mass, paths))
                .collect();
            graph_tier.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            for (graph_rank, (chunk_id, _mass, paths)) in graph_tier.into_iter().enumerate() {
                if mode == SearchMode::Graph && fused.contains_key(&chunk_id) {
                    continue; // graph mode surfaces expansion chunks only
                }
                let entry = fused.entry(chunk_id).or_insert(Fused {
                    doc_id: String::new(),
                    text: None,
                    vector_rank: None,
                    vector_score: None,
                    graph_rank: None,
                    graph_via: Vec::new(),
                });
                entry.graph_rank = Some(graph_rank);
                entry.graph_via = paths;
            }
        }
    }

    // RRF fusion, then order and truncate.
    let mut scored: Vec<(f64, HybridResult)> = fused
        .into_iter()
        .filter(|(_, f)| match mode {
            SearchMode::Graph => f.vector_rank.is_none() && f.graph_rank.is_some(),
            _ => true,
        })
        .map(|(chunk_id, f)| {
            let mut rrf = 0.0;
            let mut via = Vec::new();
            if let Some(rank) = f.vector_rank {
                rrf += 1.0 / (RRF_CONSTANT + rank as f64);
                via.push(format!(
                    "vector rank {} (score {:.4})",
                    rank + 1,
                    f.vector_score.unwrap_or_default()
                ));
            }
            if let Some(rank) = f.graph_rank {
                rrf += 1.0 / (RRF_CONSTANT + rank as f64);
                for path in &f.graph_via {
                    via.push(format!("graph rank {} via {path}", rank + 1));
                }
            }
            let source = match (f.vector_rank.is_some(), f.graph_rank.is_some()) {
                (true, true) => ResultSource::Both,
                (true, false) => ResultSource::Vector,
                (false, true) => ResultSource::Graph,
                (false, false) => unreachable!("fused entry without any stage rank"),
            };
            (
                rrf,
                HybridResult {
                    chunk_id,
                    doc_id: f.doc_id,
                    score: f.vector_score,
                    text: f.text.unwrap_or_default(),
                    source,
                    via,
                },
            )
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.chunk_id.cmp(&b.1.chunk_id))
    });
    let mut results: Vec<HybridResult> = scored.into_iter().map(|(_, r)| r).collect();
    results.truncate(k);

    // Hydrate graph-only results, which entered fusion without text.
    for result in &mut results {
        if result.text.is_empty() {
            let chunk = store.get_chunk(&result.chunk_id).await?;
            if let Some(chunk) = chunk {
                result.text = chunk.text;
                result.doc_id = chunk.doc_id;
            }
        }
    }
    Ok(results)
}
