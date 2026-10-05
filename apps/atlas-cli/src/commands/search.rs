//! `atlas search` — search learned content.
//!
//! Functionality arrives with the embedding engine (M04) and is refined by
//! hybrid retrieval (M08).

pub fn run(query: &str) -> anyhow::Result<()> {
    tracing::info!(query, "search invoked");
    println!("search: wired — search arrives with the embedding engine (M04).");
    Ok(())
}
