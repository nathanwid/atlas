//! Question answering over Atlas's own knowledge (M09).
//!
//! `atlas-qa` turns M08 hybrid retrieval into evidence-grounded answers:
//!
//! ```text
//! question → hybrid retrieval → sentence scoring → extractive answer
//!          ↘ knowledge-graph lead-in (entity matches in the question)
//! ```
//!
//! The strategy is deliberately **extractive**: answer sentences are quoted
//! verbatim from learned chunks with `[document, chunk]` citations, and a
//! confidence level is derived from retrieval strength. There is no LLM, no
//! abstractive generation, no conversation state — the answer is exactly
//! what Atlas learned, with provenance. When retrieval comes back empty or
//! nothing overlaps the question, the answer reports insufficient evidence
//! instead of fabricating content.
//!
//! Determinism: word-overlap sentence scoring, fixed tie-breaks (overlap →
//! vector score → chunk id), graph lead-ins in relationship insertion
//! order. Same store + question ⇒ identical answer.

mod answer;
mod engine;

pub use answer::{Answer, AnswerSentence, Confidence};
pub use engine::ask;
