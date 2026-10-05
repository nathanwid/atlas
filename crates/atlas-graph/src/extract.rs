//! Deterministic entity and relationship extraction.
//!
//! Two rules over sentences, in order of confidence:
//!
//! 1. **Explicit**: `Entity — predicate phrase — Entity`, where the
//!    predicate phrase is a normalized gap between two consecutive entity
//!    candidates that exactly matches a configured predicate ("uses",
//!    "depends on", …).
//! 2. **Co-occurrence**: two entities in the same sentence that did not
//!    receive an explicit link become `co_occurs_with` edges, with the
//!    smaller entity ID as source for determinism.
//!
//! Entity recognition is deliberately simple: maximal capitalized word runs
//! ("Tokio", "Knowledge Graph") plus a configurable lexicon that also
//! matches lowercase mentions ("rust"). A run that starts a sentence is
//! excluded unless it is a lexicon hit — sentence-initial capitalization is
//! grammatical, not semantic. This is a heuristic, and the graph is allowed
//! to be heuristic: tokenization must be lossless, extraction interprets.
//!
//! The whole extractor is deterministic: sentences in text order, words in
//! order, entities in first-seen order, edges via
//! [`KnowledgeGraph::link`]'s insertion-ordered vecs.

use crate::graph::{EdgeOrigin, Evidence, KnowledgeGraph, normalize};

/// Configuration for the extractor. Both lists are matched after
/// normalization, so entries can be written in any case.
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Predicate phrases that create explicit edges between two
    /// consecutive entities ("uses", "depends on", …).
    pub predicates: Vec<String>,
    /// Known entity surface forms, matched anywhere in a sentence
    /// regardless of capitalization.
    pub lexicon: Vec<String>,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        ExtractorConfig {
            predicates: vec![
                "uses",
                "provides",
                "includes",
                "extends",
                "depends on",
                "is a",
                "mentions",
                "similar to",
                "requires",
                "supports",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            lexicon: ["rust", "tokio", "cargo", "atlas", "go", "python"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Extractor {
    config: ExtractorConfig,
}

/// A word occurrence inside a sentence (identity by word index; the gap
/// between candidates is the normalized text of the words between them).
#[derive(Debug, Clone)]
struct Word {
    text: String,
}

/// An entity candidate: a display form plus the word range it covers.
#[derive(Debug, Clone)]
struct Candidate {
    display: String,
    words: (usize, usize), // [start, end) word indices
}

impl Extractor {
    pub fn new(config: ExtractorConfig) -> Self {
        Extractor { config }
    }

    /// Extract entities and relationships from one document into `graph`.
    pub fn extract(&self, doc_id: &str, text: &str, graph: &mut KnowledgeGraph) {
        for sentence in sentences(text) {
            let words = words(&sentence);
            let candidates = self.candidates(&words);
            if candidates.len() < 2 {
                // Still intern single entities so the graph knows them.
                for c in &candidates {
                    graph.intern_entity(&c.display);
                }
                continue;
            }
            let ids: Vec<u32> = candidates
                .iter()
                .map(|c| graph.intern_entity(&c.display))
                .collect();
            let evidence = Evidence {
                doc_id: doc_id.to_string(),
                sentence: sentence.clone(),
            };

            // Rule 1: explicit predicate links between consecutive candidates.
            let mut explicit_pairs: Vec<(u32, u32)> = Vec::new();
            for w in 0..candidates.len() - 1 {
                let gap = gap_text(&words, candidates[w].words.1, candidates[w + 1].words.0);
                if let Some(predicate) = self.matching_predicate(&gap) {
                    let pid = graph.intern_predicate(&predicate);
                    graph.link(
                        ids[w],
                        pid,
                        ids[w + 1],
                        EdgeOrigin::Explicit,
                        evidence.clone(),
                    );
                    explicit_pairs.push((ids[w].min(ids[w + 1]), ids[w].max(ids[w + 1])));
                }
            }

            // Rule 2: co-occurrence fallback for pairs without an explicit
            // link this sentence. Canonical order: smaller ID is the source.
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let (lo, hi) = (ids[i].min(ids[j]), ids[i].max(ids[j]));
                    if explicit_pairs.contains(&(lo, hi)) {
                        continue;
                    }
                    let pid = graph.intern_predicate("co_occurs_with");
                    graph.link(lo, pid, hi, EdgeOrigin::CoOccurrence, evidence.clone());
                }
            }
        }
    }

    fn matching_predicate(&self, gap: &str) -> Option<String> {
        let gap = normalize(gap);
        if gap.is_empty() {
            return None;
        }
        self.config
            .predicates
            .iter()
            .find(|p| normalize(p) == gap)
            .cloned()
    }

    /// Entity candidates for one sentence, in word order.
    fn candidates(&self, words: &[Word]) -> Vec<Candidate> {
        let lexicon: Vec<String> = self.config.lexicon.iter().map(|l| normalize(l)).collect();
        let mut out: Vec<Candidate> = Vec::new();

        // Maximal capitalized runs, excluding a run that starts the
        // sentence (grammatical capitalization, not semantic) unless a
        // lexicon hit.
        let mut i = 0;
        while i < words.len() {
            if is_capitalized(&words[i]) {
                let mut end = i + 1;
                while end < words.len() && is_capitalized(&words[end]) {
                    end += 1;
                }
                let starts_sentence = i == 0;
                let in_lexicon = (i..end).any(|w| lexicon.contains(&normalize(&words[w].text)));
                if !starts_sentence || in_lexicon {
                    let display = words[i..end]
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    out.push(Candidate {
                        display,
                        words: (i, end),
                    });
                }
                i = end;
            } else {
                i += 1;
            }
        }

        // Lexicon hits not already covered by a run (lowercase mentions).
        for (w, word) in words.iter().enumerate() {
            let key = normalize(&word.text);
            if lexicon.contains(&key) && !out.iter().any(|c| c.words.0 <= w && w < c.words.1) {
                out.push(Candidate {
                    display: word.text.clone(),
                    words: (w, w + 1),
                });
            }
        }
        out.sort_by_key(|c| c.words.0);
        out.dedup_by(|a, b| a.words == b.words);
        out
    }
}

fn is_capitalized(word: &Word) -> bool {
    word.text.chars().next().is_some_and(|c| c.is_uppercase())
}

/// Split into sentences at `.`, `!`, `?`, and newlines. Decimal points
/// inside numbers will split — a documented heuristic, acceptable for an
/// educational extractor.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        match c {
            '.' | '!' | '?' | '\n' => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    out.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

/// Words are runs of alphanumeric characters plus apostrophes.
fn words(sentence: &str) -> Vec<Word> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in sentence.char_indices() {
        if c.is_alphanumeric() || c == '\'' {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            out.push(Word {
                text: sentence[s..i].to_string(),
            });
        }
    }
    if let Some(s) = start {
        out.push(Word {
            text: sentence[s..].to_string(),
        });
    }
    out
}

/// The normalized text between two word offsets (exclusive), i.e. the
/// phrase that would have to be a predicate.
fn gap_text(words: &[Word], from_end: usize, to_start: usize) -> String {
    words[from_end..to_start]
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_with_default() -> KnowledgeGraph {
        KnowledgeGraph::new()
    }

    #[test]
    fn sentences_split_on_punctuation_and_newlines() {
        assert_eq!(sentences("A. B! C?\nD"), vec!["A", "B", "C", "D"]);
        assert_eq!(sentences("no terminator"), vec!["no terminator"]);
        assert!(sentences("  \n . \n").is_empty());
    }

    #[test]
    fn words_keep_offsets() {
        let w = words("Rust uses Tokio.");
        assert_eq!(w[0].text, "Rust");
        assert_eq!(w.len(), 3);
    }

    #[test]
    fn capitalized_runs_become_candidates() {
        let ex = Extractor::new(ExtractorConfig::default());
        let words = words("the Knowledge Graph engine learns");
        let candidates = ex.candidates(&words);
        let displays: Vec<&str> = candidates.iter().map(|c| c.display.as_str()).collect();
        assert_eq!(displays, vec!["Knowledge Graph"]);
    }

    #[test]
    fn sentence_initial_run_is_excluded_unless_lexicon() {
        let ex = Extractor::new(ExtractorConfig::default());
        let plain = ex.candidates(&words("Caterpillar learns things"));
        assert!(
            plain.is_empty(),
            "sentence-initial non-lexicon run must be excluded"
        );

        let with_lexicon = ex.candidates(&words("Atlas learns things"));
        assert_eq!(with_lexicon.len(), 1);
        assert_eq!(with_lexicon[0].display, "Atlas");
    }

    #[test]
    fn lexicon_matches_lowercase_mentions() {
        let ex = Extractor::new(ExtractorConfig::default());
        let candidates = ex.candidates(&words("many systems prefer rust these days"));
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].display, "rust");
    }

    #[test]
    fn explicit_pattern_creates_directed_edge() {
        let ex = Extractor::new(ExtractorConfig::default());
        let mut g = graph_with_default();
        ex.extract("doc", "Rust uses Tokio for async.", &mut g);

        assert_eq!(
            g.relationship_count(),
            1,
            "explicit pair skips co-occurrence fallback"
        );
        let rel = g.relationships().next().unwrap().1;
        assert_eq!(rel.origin, EdgeOrigin::Explicit);
        assert_eq!(g.predicate(rel.predicate), Some("uses"));
        let src = g.entity(rel.src).unwrap().display.clone();
        let dst = g.entity(rel.dst).unwrap().display.clone();
        assert_eq!(src, "Rust");
        assert_eq!(dst, "Tokio");
        assert_eq!(rel.evidence.len(), 1);
        assert_eq!(rel.evidence[0].doc_id, "doc");
        assert_eq!(rel.evidence[0].sentence, "Rust uses Tokio for async");
    }

    #[test]
    fn multi_word_predicates_match() {
        let ex = Extractor::new(ExtractorConfig::default());
        let mut g = graph_with_default();
        ex.extract("doc", "Tokio depends on Cargo heavily.", &mut g);
        let rel = g.relationships().next().unwrap().1;
        assert_eq!(g.predicate(rel.predicate), Some("depends on"));
        assert_eq!(rel.origin, EdgeOrigin::Explicit);
    }

    #[test]
    fn co_occurrence_falls_back_with_canonical_order() {
        let ex = Extractor::new(ExtractorConfig::default());
        let mut g = graph_with_default();
        // Tokio interned before Rust, so tokio < rust; canonical source must
        // be the smaller ID regardless of sentence order.
        ex.extract("doc", "Tokio and Rust shipped together.", &mut g);
        assert_eq!(g.relationship_count(), 1);
        let rel = g.relationships().next().unwrap().1;
        assert_eq!(rel.origin, EdgeOrigin::CoOccurrence);
        assert!(rel.src < rel.dst);
        assert_eq!(g.predicate(rel.predicate), Some("co_occurs_with"));
    }

    #[test]
    fn later_explicit_evidence_upgrades_co_occurrence() {
        let ex = Extractor::new(ExtractorConfig::default());
        let mut g = graph_with_default();
        ex.extract("d1", "Rust and Cargo shipped.", &mut g);
        ex.extract("d2", "Rust extends Cargo broadly.", &mut g);

        // The fallback edge keeps its co_occurs_with predicate but gains
        // Explicit origin; the pattern edge is recorded under its own
        // predicate. Both keep their evidence.
        assert_eq!(g.relationship_count(), 2);
        let co_occurs: Vec<_> = g
            .relationships()
            .filter(|(_, r)| g.predicate(r.predicate) == Some("co_occurs_with"))
            .map(|(_, r)| r)
            .collect();
        assert_eq!(co_occurs.len(), 1);
        assert_eq!(co_occurs[0].origin, EdgeOrigin::Explicit);
        assert_eq!(co_occurs[0].evidence.len(), 1);
        let explicit: Vec<_> = g
            .relationships()
            .filter(|(_, r)| g.predicate(r.predicate) == Some("extends"))
            .map(|(_, r)| r)
            .collect();
        assert_eq!(explicit.len(), 1);
        assert_eq!(explicit[0].origin, EdgeOrigin::Explicit);
    }

    #[test]
    fn pattern_and_fallback_coexist_in_one_sentence() {
        let ex = Extractor::new(ExtractorConfig::default());
        let mut g = graph_with_default();
        // Rust->Tokio explicit; Rust<->Atlas co-occurrence; Tokio<->Atlas
        // co-occurrence. Three relationships total.
        ex.extract("doc", "Rust uses Tokio for Atlas features.", &mut g);
        assert_eq!(g.relationship_count(), 3);
        let origins: Vec<EdgeOrigin> = g.relationships().map(|(_, r)| r.origin).collect();
        assert_eq!(
            origins
                .iter()
                .filter(|o| **o == EdgeOrigin::Explicit)
                .count(),
            1
        );
        assert_eq!(
            origins
                .iter()
                .filter(|o| **o == EdgeOrigin::CoOccurrence)
                .count(),
            2
        );
    }

    #[test]
    fn extraction_is_deterministic() {
        let ex = Extractor::new(ExtractorConfig::default());
        let text = "Rust uses Tokio. Atlas extends Rust with Cargo. Python and Go ship.";
        let mut a = KnowledgeGraph::new();
        ex.extract("doc", text, &mut a);
        let mut b = KnowledgeGraph::new();
        ex.extract("doc", text, &mut b);
        assert_eq!(a.entity_count(), b.entity_count());
        assert_eq!(a.relationship_count(), b.relationship_count());
        for (id, _) in a.relationships() {
            let ra = a.relationship(id).unwrap();
            let rb = b.relationship(id).unwrap();
            assert_eq!(
                (ra.src, ra.predicate, ra.dst, &ra.origin),
                (rb.src, rb.predicate, rb.dst, &rb.origin)
            );
        }
    }
}
