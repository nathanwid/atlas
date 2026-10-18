#![cfg(feature = "sqlite-backend")]
//! Bundled-SQLite version guard (storage PoC).
//!
//! SQLite 3.7.0 through 3.51.2 carried the "WAL-reset bug" (fixed in
//! 3.51.3, 2026-03-13): a rare WAL corruption race requiring two or more
//! connections to write/checkpoint at the same instant. Atlas uses a
//! single connection, so exposure is nil by usage pattern — but a bundled
//! engine in the vulnerable range is still unacceptable on principle.
//!
//! This guard checks the version of the actually-linked SQLite library at
//! runtime (`libsqlite3_sys::version()`), not the declared `rusqlite`
//! version, so a dependency change that quietly swaps in an older bundled
//! engine fails here.

#[test]
fn linked_sqlite_is_at_least_3_51_3() {
    let version = rusqlite::version();
    let mut parts = version.split('.');
    let parse = |s: Option<&str>, what: &str| {
        s.unwrap_or_else(|| panic!("malformed SQLite version {version:?}: missing {what}"))
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse::<u32>()
            .unwrap_or_else(|_| panic!("malformed SQLite version {version:?}: bad {what}"))
    };
    let (major, minor, patch) = (
        parse(parts.next(), "major"),
        parse(parts.next(), "minor"),
        parse(parts.next(), "patch"),
    );

    assert!(
        (major, minor, patch) >= (3, 51, 3),
        "linked SQLite {version} predates the WAL-reset fix (3.51.3, 2026-03-13); \
         the bundled engine must be upgraded before this backend ships"
    );
    // The PoC pins rusqlite 0.40.x, which bundles 3.53.x; anything lower
    // than 3.53 means the dependency drifted, even if still patched.
    assert!(
        (major, minor, patch) >= (3, 53, 0),
        "linked SQLite {version} is patched but older than the 3.53.x the \
         PoC pins (rusqlite 0.40.x) — dependency drift, investigate"
    );
}
