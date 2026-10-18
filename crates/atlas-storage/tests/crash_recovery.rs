#![cfg(feature = "sqlite-backend")]
//! Process-crash recovery for the SQLite backend (storage PoC).
//!
//! A child process opens the durable SQLite store, commits exactly one
//! known write stage (the document row), reports the stage via a marker
//! file, and parks. The parent polls for the marker â€” a synchronization
//! point guaranteeing the kill lands *after* the document commit and
//! *before* any chunk/embedding write â€” then force-kills the child
//! (`TerminateProcess` on Windows, SIGKILL on Linux; no graceful drop, no
//! SQLite close). The parent reopens the store, asserts the known partial
//! state survived, and verifies the documented re-`learn` repair.
//!
//! What this proves and what it does not:
//!
//! * Proven: a committed transaction survives an unclean process
//!   termination and the store reopens to a consistent state (WAL
//!   recovery), matching Atlas's repair contract.
//! * Not proven: power loss / OS crash. That guarantee comes from
//!   `synchronous=FULL` fsync-on-commit per SQLite's upstream
//!   documentation, but this test cannot cut power, and we do not claim
//!   its result as evidence for it.

use atlas_embedding::Embedder;
use atlas_storage::backend::Backend;
use atlas_storage::{SqliteStore, Store, doc_id_for, learn};
use atlas_tokenizer::BpeTokenizer;
use std::time::{Duration, Instant};

struct FakeEmbedder;

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        "crash-test-model"
    }

    fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| vec![t.len() as f32, t.starts_with('a') as u8 as f32])
            .collect())
    }
}

fn tokenizer() -> BpeTokenizer {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
    );
    let f = std::fs::File::open(path).expect("ranks fixture");
    BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("ranks parse")
}

/// Child entry point. A no-op in the parent's normal test run; with
/// `ATLAS_CRASH_CHILD=1` it performs the scripted write stage and parks
/// until killed. Runs as the `--exact crash_child_worker` filter so the
/// harness executes nothing else in the child process.
#[test]
fn crash_child_worker() {
    let Ok(role) = std::env::var("ATLAS_CRASH_CHILD") else {
        return;
    };
    let db = std::env::var("ATLAS_CRASH_DB").expect("db path");
    let marker = std::env::var("ATLAS_CRASH_MARKER").expect("marker path");
    let text = std::env::var("ATLAS_CRASH_TEXT").expect("text");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        match role.as_str() {
            "sqlite" => {
                let store = SqliteStore::open_durable(&db).await.unwrap();
                // Deterministic crash stage: exactly the document row,
                // committed durably (synchronous=FULL), nothing else.
                store
                    .upsert_document(&doc_id_for(&text), &text)
                    .await
                    .unwrap();
            }
            "surrealkv" => {
                let store = Store::open_durable(&db).await.unwrap();
                store
                    .upsert_document(&doc_id_for(&text), &text)
                    .await
                    .unwrap();
            }
            // Full grouped learn: with transaction batching the whole
            // document lands in one commit, so the marker means the
            // group has committed (or learn errored before it).
            "sqlite_batch" => {
                let store = SqliteStore::open_durable(&db).await.unwrap();
                learn(&store, &tokenizer(), &FakeEmbedder, &text)
                    .await
                    .unwrap();
            }
            other => panic!("unknown crash-child role {other}"),
        }
    });
    // Signaled only after the stage above has returned Ok.
    std::fs::write(&marker, b"stage-committed").expect("marker write");
    loop {
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Spawn the crash child for `role`, wait for its stage marker, kill it.
fn spawn_and_kill(
    role: &str,
    db: &std::path::Path,
    marker: &std::path::Path,
    text: &str,
) -> anyhow::Result<()> {
    let exe = std::env::current_exe().expect("test binary path");
    let mut child = std::process::Command::new(exe)
        .args(["--exact", "crash_child_worker", "--nocapture"])
        .env("ATLAS_CRASH_CHILD", role)
        .env("ATLAS_CRASH_DB", db)
        .env("ATLAS_CRASH_MARKER", marker)
        .env("ATLAS_CRASH_TEXT", text)
        .spawn()
        .expect("spawn crash child");

    let mut failure: Option<String> = None;
    let deadline = Instant::now() + Duration::from_secs(60);
    while !marker.exists() {
        if deadline <= Instant::now() {
            failure = Some("crash child never reached its write stage".into());
            break;
        }
        if matches!(child.try_wait(), Ok(Some(_))) {
            failure = Some("crash child exited before writing its stage marker".into());
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    // Unclean termination: no Drop, no SQLite close, no fsync by the child.
    // Always reaped, including early-failure paths.
    child.kill().expect("kill crash child");
    child.wait().expect("reap crash child");
    if let Some(message) = failure {
        panic!("{message}");
    }
    Ok(())
}

#[tokio::test]
async fn process_kill_between_document_and_chunks_is_repaired() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("atlas.db");
    let marker = dir.path().join("marker");
    let text = "alpha beta gamma delta epsilon zeta eta theta";
    let doc_id = doc_id_for(text);

    spawn_and_kill("sqlite", &db, &marker, text).unwrap();

    // The committed document row survived the kill...
    let store = SqliteStore::open_durable(&db).await.unwrap();
    assert!(
        store.document_exists(&doc_id).await.unwrap(),
        "committed document row must survive an unclean kill"
    );
    // ...at exactly the known stage: no chunks were written.
    assert!(
        store
            .get_chunk(&format!("{doc_id}:0"))
            .await
            .unwrap()
            .is_none(),
        "the kill must land before any chunk write"
    );

    // The documented repair: re-running learn completes the document.
    let summary = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    assert!(
        !summary.duplicate,
        "partial state is not a healthy duplicate"
    );
    assert_eq!(summary.chunks, 1);
    assert!(
        store
            .get_chunk(&format!("{doc_id}:0"))
            .await
            .unwrap()
            .is_some()
    );

    // And the repaired document is a clean duplicate afterwards.
    let again = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    assert!(again.duplicate);
}

/// Diagnostic companion (run explicitly:
/// `cargo test -p atlas-storage --features sqlite-backend -- --ignored --exact surrealkv_kill_recovery`):
/// the same kill-at-known-stage harness against the SurrealKV durable
/// engine, answering empirically whether its committed document row
/// survives an unclean process termination in the vendored 0.1.6 build.
/// Not part of the default gate because its outcome is a finding, not a
/// regression assertion.
#[tokio::test]
#[ignore = "diagnostic: reports SurrealKV 0.1.6 crash behavior; run explicitly"]
async fn surrealkv_kill_recovery_diagnostic() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("surreal");
    let marker = dir.path().join("marker");
    let text = "alpha beta gamma delta epsilon zeta eta theta";
    let doc_id = doc_id_for(text);

    spawn_and_kill("surrealkv", &db, &marker, text).unwrap();

    let store = Store::open_durable(&db).await.unwrap();
    let survived = store.document_exists(&doc_id).await.unwrap();
    let chunk_there = store
        .get_chunk(&format!("{doc_id}:0"))
        .await
        .unwrap()
        .is_some();
    let summary = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    eprintln!(
        "surrealkv diagnostic: document survived kill = {survived}; \
         unexpected chunk present = {chunk_there}; re-learn duplicate = {}",
        summary.duplicate
    );
    assert!(
        survived,
        "committed document row did not survive the kill — durability finding"
    );
}

/// Interrupted ingestion, batching edition: the crash child runs a full
/// grouped `learn` (one atomic transaction per document), so a kill after
/// the group commits leaves the complete document — never a partial
/// subset. All-or-nothing is asserted, and both outcomes are repaired by
/// the documented re-`learn`.
#[tokio::test]
async fn killed_mid_learn_leaves_all_or_nothing_state() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("atlas.db");
    let marker = dir.path().join("marker");
    let text = "Rust requires PyO3 for bindings.";
    let doc_id = doc_id_for(text);

    spawn_and_kill("sqlite_batch", &db, &marker, text).unwrap();

    let store = SqliteStore::open_durable(&db).await.unwrap();
    let committed = store.document_exists(&doc_id).await.unwrap();
    let chunk_there = store
        .get_chunk(&format!("{doc_id}:0"))
        .await
        .unwrap()
        .is_some();
    // All-or-nothing: document and its chunk must agree.
    assert_eq!(
        committed, chunk_there,
        "atomic group: document and chunk must not diverge after a kill"
    );

    // Either outcome is repaired by the documented re-learn.
    let summary = learn(&store, &tokenizer(), &FakeEmbedder, text)
        .await
        .unwrap();
    assert!(
        summary.duplicate == committed,
        "re-learn must be a duplicate iff the group had committed"
    );
    assert!(
        store
            .get_chunk(&format!("{doc_id}:0"))
            .await
            .unwrap()
            .is_some()
    );
}
