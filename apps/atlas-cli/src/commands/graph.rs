//! `atlas graph` — the relationship graph around an entity.
//!
//! Functionality arrives with M08 hybrid retrieval, over the persisted graph (M06/M07).

pub async fn run(entity: &str) -> anyhow::Result<()> {
    tracing::info!(entity, "graph invoked");
    println!("graph: wired — the relationship graph arrives with M06.");
    Ok(())
}
