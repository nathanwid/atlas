//! File publication shared by vocabulary and model checkpoints.
use std::{io, path::Path};
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_with(path, bytes, || Ok(()))
}
fn write_with(
    path: &Path,
    bytes: &[u8],
    before_replace: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::other("checkpoint path has no file name"))?;
    let (temporary, mut file) = loop {
        let mut name = name.to_os_string();
        name.push(format!(
            ".atlas-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let temporary = parent.join(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    // Remove an unpublished temporary file on every normal error path.
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(temporary.clone());
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    before_replace()?;
    // Same-directory rename replaces the destination atomically on supported
    // local filesystems, including Windows. Never remove the old target first.
    std::fs::rename(&temporary, path)?;
    #[cfg(unix)]
    std::fs::File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failure_before_publication_preserves_previous_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint");
        std::fs::write(&path, b"old").unwrap();
        assert!(write_with(&path, b"new", || Err(io::Error::other("injected failure"))).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
    }
    #[test]
    fn replacement_publishes_complete_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint");
        write_atomic(&path, b"old").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }
}
