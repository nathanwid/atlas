//! `atlas learn` — learning from text.
//!
//! Functionality arrives with the vocabulary (M03) and embedding (M04)
//! milestones: learned text becomes chunks, then embeddings, then knowledge.

pub fn run(text: &str) -> anyhow::Result<()> {
    tracing::info!(text, "learn invoked");
    println!(
        "learn: wired — learning from text arrives with the vocabulary and embedding milestones (M03/M04)."
    );
    Ok(())
}
