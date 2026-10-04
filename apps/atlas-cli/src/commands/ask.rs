//! `atlas ask` — question answering over Atlas's own knowledge.
//!
//! Functionality arrives with question answering (M09), on top of search and
//! the knowledge graph.

pub fn run(question: &str) -> anyhow::Result<()> {
    tracing::info!(question, "ask invoked");
    println!("ask: wired — question answering arrives with M09.");
    Ok(())
}
