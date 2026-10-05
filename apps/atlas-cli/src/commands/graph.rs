//! `atlas graph` — the relationship graph around an entity.
//!
//! Functionality arrives with the knowledge graph (M06).

pub fn run(entity: &str) -> anyhow::Result<()> {
    tracing::info!(entity, "graph invoked");
    println!("graph: wired — the relationship graph arrives with M06.");
    Ok(())
}
