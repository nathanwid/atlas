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
            out.push_str(&escape_terminal_controls(lead));
            out.push('\n');
        }
        out.push_str("From the learned content:\n");
        for (i, sentence) in self.sentences.iter().enumerate() {
            out.push_str(&format!(
                "{}. \"{}\" [doc {}, chunk {}]\n",
                i + 1,
                escape_terminal_controls(&sentence.text),
                escape_terminal_controls(&sentence.doc_id),
                escape_terminal_controls(&sentence.chunk_id)
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

fn escape_terminal_controls(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_control() {
            escaped.push_str(&format!("\\u{{{:x}}}", ch as u32));
        } else {
            escaped.push(ch);
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendering_does_not_emit_controls_from_learned_evidence() {
        let answer = Answer {
            question: "What uses\u{1b}[2J Tokio?".into(),
            lead_in: Some("Rust -[uses\u{7}]-> Tokio".into()),
            sentences: vec![AnswerSentence {
                text: "Rust\u{1b}[31m uses Tokio\u{9b}2J".into(),
                doc_id: "doc\u{7}".into(),
                chunk_id: "chunk\u{1b}[K".into(),
                score: Some(0.9),
            }],
            confidence: Confidence::High,
            insufficient: false,
            multi_source: false,
        };

        let rendered = answer.render();

        assert!(rendered.contains("Rust\\u{1b}[31m uses Tokio\\u{9b}2J"));
        assert!(rendered.contains("Rust -[uses\\u{7}]-> Tokio"));
        assert!(rendered.contains("doc\\u{7}"));
        assert!(rendered.contains("chunk\\u{1b}[K"));
        assert!(rendered.chars().all(|ch| !ch.is_control() || ch == '\n'));
        assert!(answer.sentences[0].text.contains('\u{1b}'));
    }
}
