//! The knowledge graph: entities, directed relationships, and traversal.
//!
//! Ownership boundary: this crate is self-contained (std only). Persistence
//! is M07; graph scoring and personalized walks are M08; entity linking to
//! vocabulary IDs or embeddings is deliberately out of scope — entities are
//! surface terms, not tokens.
//!
//! Determinism: entity/predicate/relationship IDs are dense and assigned in
//! first-seen order; every query output is derived from insertion-ordered
//! vectors, never from hash-map iteration. Hash maps exist only for lookup.

use std::collections::{HashMap, VecDeque};
use std::fmt;

/// A vocabulary entry's — an entity's — stable numeric identity.
pub type EntityId = u32;
/// Interned predicate string identity.
pub type PredicateId = u32;
/// A relationship's stable numeric identity.
pub type RelationshipId = u32;

/// How a relationship was learned. `Explicit` comes from a recognized
/// predicate pattern ("Rust uses Tokio"); `CoOccurrence` is the
/// lower-confidence fallback when two entities merely share a sentence.
/// `Explicit` is strictly stronger: evidence upgrades CoOccurrence to
/// Explicit, never the reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeOrigin {
    Explicit,
    CoOccurrence,
}

impl fmt::Display for EdgeOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeOrigin::Explicit => f.write_str("explicit"),
            EdgeOrigin::CoOccurrence => f.write_str("co-occurrence"),
        }
    }
}

/// An entity: a surface term with its first-seen display form plus an
/// optional type label (extraction never sets `kind`; callers may).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entity {
    pub display: String,
    pub kind: Option<String>,
}

/// Where a relationship was seen: which document and which sentence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Evidence {
    pub doc_id: String,
    pub sentence: String,
}

/// A directed, labeled relationship between two entities.
#[derive(Debug, Clone)]
pub struct Relationship {
    pub src: EntityId,
    pub predicate: PredicateId,
    pub dst: EntityId,
    pub origin: EdgeOrigin,
    pub evidence: Vec<Evidence>,
}

/// One hop from an entity, with enough context to be useful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Neighbor {
    pub relationship: RelationshipId,
    pub other: EntityId,
    pub direction: Direction,
    pub predicate: PredicateId,
    pub origin: EdgeOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Outgoing,
    Incoming,
}

/// Normalize a surface term into its identity key: trimmed, lowercased.
/// This is the consumer-layer normalization decision the project deferred
/// until a milestone needed it; the graph is that milestone.
pub fn normalize(term: &str) -> String {
    term.trim().to_lowercase()
}

#[derive(Debug, Default)]
pub struct KnowledgeGraph {
    entities: Vec<Entity>,
    entity_ids: HashMap<String, EntityId>,
    predicates: Vec<String>,
    predicate_ids: HashMap<String, PredicateId>,
    edges: Vec<Relationship>,
    edge_ids: HashMap<(EntityId, PredicateId, EntityId), RelationshipId>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entities / relationships / distinct predicates.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn relationship_count(&self) -> usize {
        self.edges.len()
    }

    pub fn predicate_count(&self) -> usize {
        self.predicates.len()
    }

    /// Look up an entity by any surface form (normalized before lookup).
    pub fn find_entity(&self, term: &str) -> Option<EntityId> {
        self.entity_ids.get(&normalize(term)).copied()
    }

    /// The entity for an ID.
    pub fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(id as usize)
    }

    /// The predicate string for an ID.
    pub fn predicate(&self, id: PredicateId) -> Option<&str> {
        self.predicates.get(id as usize).map(String::as_str)
    }

    /// Look up a predicate ID by surface form (normalized before lookup).
    pub fn find_predicate(&self, predicate: &str) -> Option<PredicateId> {
        self.predicate_ids.get(&normalize(predicate)).copied()
    }

    /// The normalized identity key of an entity.
    pub fn entity_key(&self, id: EntityId) -> Option<String> {
        self.entities
            .get(id as usize)
            .map(|e| normalize(&e.display))
    }

    /// All predicates, in insertion order.
    pub fn predicates(&self) -> impl Iterator<Item = (PredicateId, &str)> {
        self.predicates
            .iter()
            .enumerate()
            .map(|(i, p)| (i as PredicateId, p.as_str()))
    }

    /// The relationship for an ID.
    pub fn relationship(&self, id: RelationshipId) -> Option<&Relationship> {
        self.edges.get(id as usize)
    }

    /// All relationships, in insertion order.
    pub fn relationships(&self) -> impl Iterator<Item = (RelationshipId, &Relationship)> {
        self.edges
            .iter()
            .enumerate()
            .map(|(i, r)| (i as RelationshipId, r))
    }

    /// All entities, in insertion order.
    pub fn entities(&self) -> impl Iterator<Item = (EntityId, &Entity)> {
        self.entities
            .iter()
            .enumerate()
            .map(|(i, e)| (i as EntityId, e))
    }

    /// Attach a type label to an entity. Extraction leaves `kind` unset;
    /// this is the caller-facing way to label entities.
    pub fn set_kind(&mut self, id: EntityId, kind: &str) -> bool {
        match self.entities.get_mut(id as usize) {
            Some(entity) => {
                entity.kind = Some(kind.to_string());
                true
            }
            None => false,
        }
    }

    /// One hop from `id` in both directions, in edge insertion order.
    pub fn neighbors(&self, id: EntityId) -> Vec<Neighbor> {
        let mut out = Vec::new();
        for (rid, rel) in self.edges.iter().enumerate() {
            if rel.src == id {
                out.push(Neighbor {
                    relationship: rid as RelationshipId,
                    other: rel.dst,
                    direction: Direction::Outgoing,
                    predicate: rel.predicate,
                    origin: rel.origin,
                });
            } else if rel.dst == id {
                out.push(Neighbor {
                    relationship: rid as RelationshipId,
                    other: rel.src,
                    direction: Direction::Incoming,
                    predicate: rel.predicate,
                    origin: rel.origin,
                });
            }
        }
        out
    }

    /// Breadth-first reachability from `id`, up to `max_hops` edges
    /// (`usize::MAX` = unlimited). Start entity excluded. Expansion follows
    /// [`KnowledgeGraph::neighbors`] order, so the output is deterministic.
    pub fn reachable(&self, id: EntityId, max_hops: usize) -> Vec<EntityId> {
        let mut in_set = vec![false; self.entities.len()];
        let mut order = Vec::new();
        let mut queue = VecDeque::new();

        in_set[id as usize] = true;
        queue.push_back((id, 0usize));
        while let Some((current, depth)) = queue.pop_front() {
            if depth == max_hops {
                continue;
            }
            for neighbor in self.neighbors(current) {
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

    /// Intern an entity by display form; identity is the normalized key.
    ///
    /// Public since M07: durable storage reconstructs graphs through these
    /// builders, which enforce the dedup/upgrade/evidence rules during
    /// rebuild.
    pub fn intern_entity(&mut self, display: &str) -> EntityId {
        let key = normalize(display);
        if let Some(&id) = self.entity_ids.get(&key) {
            return id;
        }
        let id = self.entities.len() as EntityId;
        self.entities.push(Entity {
            display: display.trim().to_string(),
            kind: None,
        });
        self.entity_ids.insert(key, id);
        id
    }

    pub fn intern_predicate(&mut self, predicate: &str) -> PredicateId {
        let key = normalize(predicate);
        if let Some(&id) = self.predicate_ids.get(&key) {
            return id;
        }
        let id = self.predicates.len() as PredicateId;
        self.predicates.push(key.clone());
        self.predicate_ids.insert(key, id);
        id
    }

    /// Record a sighting of `(src, predicate, dst)`.
    ///
    /// Dedup: one relationship per triple. Evidence merges (identical
    /// evidence is not duplicated). Origin upgrades CoOccurrence → Explicit
    /// when the stronger rule fires; Explicit never downgrades. An Explicit
    /// sighting also upgrades any existing `CoOccurrence` edge between the
    /// same two entities (either direction): explicit evidence raises the
    /// confidence of the pair's fallback relationship. Public since M07
    /// (storage reconstruction); see [`KnowledgeGraph::intern_entity`].
    pub fn link(
        &mut self,
        src: EntityId,
        predicate: PredicateId,
        dst: EntityId,
        origin: EdgeOrigin,
        evidence: Evidence,
    ) -> RelationshipId {
        if let Some(&rid) = self.edge_ids.get(&(src, predicate, dst)) {
            let rel = &mut self.edges[rid as usize];
            if !rel.evidence.contains(&evidence) {
                rel.evidence.push(evidence);
            }
            if rel.origin == EdgeOrigin::CoOccurrence && origin == EdgeOrigin::Explicit {
                rel.origin = EdgeOrigin::Explicit;
            }
            return rid;
        }
        let rid = self.edges.len() as RelationshipId;
        self.edges.push(Relationship {
            src,
            predicate,
            dst,
            origin,
            evidence: vec![evidence],
        });
        self.edge_ids.insert((src, predicate, dst), rid);
        if origin == EdgeOrigin::Explicit {
            // Confidence upgrade: the fallback edges of this entity pair are
            // no longer mere co-occurrences.
            for rel in &mut self.edges {
                if rel.origin == EdgeOrigin::CoOccurrence
                    && ((rel.src == src && rel.dst == dst) || (rel.src == dst && rel.dst == src))
                {
                    rel.origin = EdgeOrigin::Explicit;
                }
            }
        }
        rid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_dedups_triples_and_merges_evidence() {
        let mut g = KnowledgeGraph::new();
        let a = g.intern_entity("Rust");
        let b = g.intern_entity("Tokio");
        let p = g.intern_predicate("uses");
        let rid = g.link(
            a,
            p,
            b,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d1".into(),
                sentence: "Rust uses Tokio.".into(),
            },
        );
        g.link(
            a,
            p,
            b,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d2".into(),
                sentence: "Rust uses Tokio for async".into(),
            },
        );
        // Identical evidence is not duplicated.
        g.link(
            a,
            p,
            b,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d1".into(),
                sentence: "Rust uses Tokio.".into(),
            },
        );
        assert_eq!(g.relationship_count(), 1);
        assert_eq!(g.relationship(rid).unwrap().evidence.len(), 2);
    }

    #[test]
    fn origin_upgrades_to_explicit_only() {
        let mut g = KnowledgeGraph::new();
        let a = g.intern_entity("Rust");
        let b = g.intern_entity("Cargo");
        let p = g.intern_predicate("mentions");
        let rid = g.link(
            a,
            p,
            b,
            EdgeOrigin::CoOccurrence,
            Evidence {
                doc_id: "d".into(),
                sentence: "Rust and Cargo".into(),
            },
        );
        g.link(
            a,
            p,
            b,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "Rust mentions Cargo".into(),
            },
        );
        assert_eq!(g.relationship(rid).unwrap().origin, EdgeOrigin::Explicit);
        // Explicit never downgrades.
        g.link(
            a,
            p,
            b,
            EdgeOrigin::CoOccurrence,
            Evidence {
                doc_id: "e".into(),
                sentence: "Rust with Cargo".into(),
            },
        );
        assert_eq!(g.relationship(rid).unwrap().origin, EdgeOrigin::Explicit);
    }

    #[test]
    fn direction_matters() {
        let mut g = KnowledgeGraph::new();
        let a = g.intern_entity("Rust");
        let b = g.intern_entity("Tokio");
        let p = g.intern_predicate("uses");
        g.link(
            a,
            p,
            b,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "Rust uses Tokio.".into(),
            },
        );
        g.link(
            b,
            p,
            a,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "Tokio uses Rust.".into(),
            },
        );
        assert_eq!(g.relationship_count(), 2);
    }

    #[test]
    fn neighbors_report_both_directions_in_insertion_order() {
        let mut g = KnowledgeGraph::new();
        let a = g.intern_entity("A");
        let b = g.intern_entity("B");
        let c = g.intern_entity("C");
        let p = g.intern_predicate("mentions");
        g.link(
            b,
            p,
            a,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );
        g.link(
            a,
            p,
            c,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );
        let neighbors = g.neighbors(a);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].direction, Direction::Incoming);
        assert_eq!(neighbors[0].other, b);
        assert_eq!(neighbors[1].direction, Direction::Outgoing);
        assert_eq!(neighbors[1].other, c);
    }

    #[test]
    fn reachable_bfs_is_deterministic_and_hop_limited() {
        let mut g = KnowledgeGraph::new();
        let ids: Vec<EntityId> = ["A", "B", "C", "D"]
            .iter()
            .map(|n| g.intern_entity(n))
            .collect();
        let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
        let p = g.intern_predicate("mentions");
        // a -> b -> c ; b -> d ; c -> a (cycle)
        g.link(
            a,
            p,
            b,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );
        g.link(
            b,
            p,
            c,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );
        g.link(
            b,
            p,
            d,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );
        g.link(
            c,
            p,
            a,
            EdgeOrigin::Explicit,
            Evidence {
                doc_id: "d".into(),
                sentence: "s".into(),
            },
        );

        let one = g.reachable(a, 1);
        // Incoming edges count as a hop: the cycle edge c->a makes c one
        // step from a.
        assert_eq!(one, vec![b, c]);
        let all = g.reachable(a, usize::MAX);
        assert_eq!(
            all,
            vec![b, c, d],
            "cycle must not loop; order follows edge insertion"
        );
    }

    #[test]
    fn find_entity_normalizes_lookup() {
        let mut g = KnowledgeGraph::new();
        let id = g.intern_entity("Knowledge Graph");
        assert_eq!(g.find_entity("knowledge graph"), Some(id));
        assert_eq!(g.find_entity("  KNOWLEDGE GRAPH "), Some(id));
        assert_eq!(g.find_entity("something else"), None);
        // Display keeps the first-seen casing.
        assert_eq!(g.entity(id).unwrap().display, "Knowledge Graph");
    }

    #[test]
    fn set_kind_labels_entities() {
        let mut g = KnowledgeGraph::new();
        let id = g.intern_entity("Rust");
        assert_eq!(g.entity(id).unwrap().kind, None);
        assert!(g.set_kind(id, "language"));
        assert_eq!(g.entity(id).unwrap().kind.as_deref(), Some("language"));
        assert!(!g.set_kind(999, "language"));
    }

    #[test]
    fn interning_predicates_deduplicates() {
        // Regression: intern_predicate once never inserted into the lookup
        // map, so repeated interning pushed duplicate rows and lookups
        // always failed. Found by M07 graph reconstruction.
        let mut g = KnowledgeGraph::new();
        let a = g.intern_predicate("uses");
        let b = g.intern_predicate("uses");
        let c = g.intern_predicate("co_occurs_with");
        let d = g.find_predicate("USES");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(d, Some(a));
        assert_eq!(g.predicate_count(), 2, "no duplicate predicate rows");
    }
}
