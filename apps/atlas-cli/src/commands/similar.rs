//! `atlas similar` — related entities for a term.
//!
//! Functionality arrives with the embedding engine (M04): similarity over
//! embedded content.

pub fn run(term: &str) -> anyhow::Result<()> {
    tracing::info!(term, "similar invoked");
    println!("similar: wired — similarity arrives with the embedding engine (M04).");
    Ok(())
}
