//! Answer types: citations, confidence, and the insufficiency flag.

/// One extracted answer sentence with its full provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct AnswerSentence {
    /// Verbatim sentence from a learned chunk.
    pub text: String,
    /// Document the chunk belongs to.
    pub doc_id: String,
    /// Chunk the sentence was extracted from.
    pub chunk_id: String,
    /// Cosine score of the chunk, when the vector stage surfaced it.
    pub score: Option<f32>,
}

/// How strongly the retrieved evidence supports the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Strong vector score and clear question overlap.
    High,
    /// Relevant but weaker evidence.
    Medium,
    /// Marginal overlap only.
    Low,
}

/// The answer to a question.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub question: String,
    /// Optional graph lead-in derived from relationship-shaped questions
    /// (e.g. "From the knowledge graph: Rust -[uses]-> Tokio.").
    pub lead_in: Option<String>,
    /// Extracted evidence sentences, best first, each cited.
    pub sentences: Vec<AnswerSentence>,
    pub confidence: Confidence,
    /// True when retrieval found nothing on-topic. `sentences` is empty and
    /// the caller should say so rather than guess.
    pub insufficient: bool,
    /// True when evidence sentences come from more than one document.
    pub multi_source: bool,
}

impl Answer {
    /// Render the answer as the CLI prints it.
    pub fn render(&self) -> String {
        if self.insufficient {
            return format!(
                "No sufficient evidence found for: {:?}. Learn more with `atlas learn <text>`.",
                self.question
            );
        }
        let mut out = String::new();
        if let Some(lead) = &self.lead_in {
            out.push_str(lead);
            out.push('\n');
        }
        out.push_str("From the learned content:\n");
        for (i, sentence) in self.sentences.iter().enumerate() {
            out.push_str(&format!(
                "{}. \"{}\" [doc {}, chunk {}]\n",
                i + 1,
                sentence.text,
                &sentence.doc_id[..8.min(sentence.doc_id.len())],
                &sentence.chunk_id[..8.min(sentence.chunk_id.len())]
            ));
        }
        let confidence = match self.confidence {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
        };
        out.push_str(&format!("Confidence: {confidence}."));
        if self.multi_source {
            out.push_str(" Evidence spans multiple documents; sentences are quoted verbatim.");
        }
        out
    }
}
