//! `atlas search` — search persisted content (M07 vector, M08 hybrid).
//!
//! query → embed → persisted embeddings / graph expansion → ranked chunks.

use atlas_core::Config;

pub async fn run(
    config: &Config,
    query: &str,
    k: usize,
    mode: crate::cli::SearchModeArg,
) -> anyhow::Result<()> {
    let store = atlas_storage::SqliteStore::open_durable(config.data_dir.join("atlas.db")).await?;
    let embedder = atlas_embedding::FastembedEmbedder::bge_small_en_v1_5()?;

    let mode = match mode {
        crate::cli::SearchModeArg::Vector => atlas_storage::SearchMode::Vector,
        crate::cli::SearchModeArg::Hybrid => atlas_storage::SearchMode::Hybrid,
        crate::cli::SearchModeArg::Graph => atlas_storage::SearchMode::Graph,
    };
    let results =
        atlas_storage::hybrid_search(&store, &embedder, query, k, &Default::default(), mode)
            .await?;

    if results.is_empty() {
        if mode == atlas_storage::SearchMode::Graph {
            println!("search: no graph-expanded results for this query.");
        } else {
            println!("search: nothing learned yet — run `atlas learn <text>` first.");
        }
        return Ok(());
    }
    for (rank, hit) in results.iter().enumerate() {
        println!("{}", render_hit(rank, hit));
    }
    Ok(())
}

fn render_hit(rank: usize, hit: &atlas_storage::HybridResult) -> String {
    let score = hit
        .score
        .map(|s| format!("{s:.4}"))
        .unwrap_or_else(|| "-".to_string());
    let mut output = format!(
        "#{rank} source={} score={score} chunk={} doc={}",
        hit.source,
        escape_terminal_controls(&hit.chunk_id),
        escape_terminal_controls(&hit.doc_id)
    );
    for via in &hit.via {
        output.push_str(&format!("\n  via {}", escape_terminal_controls(via)));
    }
    output.push_str(&format!(
        "\n  {}",
        escape_terminal_controls(&hit.text.replace('\n', " "))
    ));
    output
}

pub(super) fn escape_terminal_controls(text: &str) -> String {
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
    fn search_rendering_does_not_emit_learned_terminal_controls() {
        let hit = atlas_storage::HybridResult {
            chunk_id: "chunk\u{1b}[2J".into(),
            doc_id: "doc\u{7}".into(),
            score: Some(0.5),
            text: "Rust\u{1b}[31m\nuses Tokio\u{9b}2J".into(),
            source: atlas_storage::ResultSource::Vector,
            via: vec!["vector\u{1b}[K".into()],
        };

        let rendered = render_hit(0, &hit);

        assert!(rendered.contains("chunk\\u{1b}[2J"));
        assert!(rendered.contains("Rust\\u{1b}[31m uses Tokio\\u{9b}2J"));
        assert!(rendered.contains("via vector\\u{1b}[K"));
        assert!(rendered.contains("doc\\u{7}"));
        assert!(rendered.chars().all(|ch| !ch.is_control() || ch == '\n'));
    }
}
