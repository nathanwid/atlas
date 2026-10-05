//! The vocabulary type: token bytes to stable dense IDs, both directions,
//! plus persistence.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A vocabulary entry's stable numeric identity.
pub type TokenId = u32;

/// Errors from loading or saving a vocabulary.
#[derive(Debug, thiserror::Error)]
pub enum VocabError {
    /// File I/O failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The file exists but is not a usable vocabulary.
    #[error("vocabulary file is corrupt: {0}")]
    Format(String),
}

#[derive(Debug, Clone, Default)]
pub struct Vocab {
    /// token bytes -> id.
    ids: HashMap<Box<[u8]>, TokenId>,
    /// id -> token bytes; `tokens[id]` is the token with that id, so ids
    /// are dense by construction.
    tokens: Vec<Box<[u8]>>,
}

/// On-disk format. `tokens[id]` is the token for `id`; order carries the
/// identity, so the file is exactly the id -> token table.
#[derive(Serialize, Deserialize)]
struct VocabFile {
    version: u32,
    tokens: Vec<Vec<u8>>,
}

const FORMAT_VERSION: u32 = 1;

impl Vocab {
    /// Look up the ID of a token.
    pub fn get(&self, token: &[u8]) -> Option<TokenId> {
        self.ids.get(token).copied()
    }

    /// Look up the token for an ID.
    pub fn token(&self, id: TokenId) -> Option<&[u8]> {
        self.tokens.get(id as usize).map(|t| &t[..])
    }

    /// Return the ID of `token`, assigning the next dense ID on first sight.
    ///
    /// This is the growth path for incremental learning: call it over
    /// tokenized input and the vocabulary grows deterministically in
    /// first-occurrence order.
    pub fn get_or_insert(&mut self, token: &[u8]) -> TokenId {
        let next = self.tokens.len() as TokenId;
        match self.ids.entry(token.into()) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(next);
                self.tokens.push(token.into());
                next
            }
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Write the vocabulary to `path` in the versioned bincode format.
    pub fn save(&self, path: &Path) -> Result<(), VocabError> {
        let file = VocabFile {
            version: FORMAT_VERSION,
            tokens: self.tokens.iter().map(|t| t.to_vec()).collect(),
        };
        let bytes = bincode::serialize(&file).expect("VocabFile is always serializable");
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a vocabulary written by [`Vocab::save`]. A corrupt, empty, or
    /// wrong-version file is an error, never a panic or a silent empty
    /// vocabulary: IDs must be stable across runs, and silently starting
    /// over would violate that.
    pub fn load(path: &Path) -> Result<Self, VocabError> {
        let bytes = std::fs::read(path)?;
        let file: VocabFile =
            bincode::deserialize(&bytes).map_err(|e| VocabError::Format(e.to_string()))?;
        if file.version != FORMAT_VERSION {
            return Err(VocabError::Format(format!(
                "unsupported version {} (expected {})",
                file.version, FORMAT_VERSION
            )));
        }
        let mut vocab = Vocab::default();
        for token in file.tokens {
            if vocab.ids.contains_key(&token[..]) {
                return Err(VocabError::Format(format!("duplicate token {token:?}")));
            }
            vocab.get_or_insert(&token);
        }
        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_or_insert_assigns_dense_sequential_ids() {
        let mut vocab = Vocab::default();
        assert_eq!(vocab.get_or_insert(b"rust"), 0);
        assert_eq!(vocab.get_or_insert(b"tokio"), 1);
        assert_eq!(vocab.get_or_insert(b"rust"), 0, "second sight is stable");
        assert_eq!(vocab.get_or_insert(b"cargo"), 2);
        assert_eq!(vocab.len(), 3);
    }

    #[test]
    fn lookups_work_both_ways() {
        let mut vocab = Vocab::default();
        let id = vocab.get_or_insert("café".as_bytes());
        assert_eq!(vocab.get("café".as_bytes()), Some(id));
        assert_eq!(vocab.token(id), Some("café".as_bytes()));
    }

    #[test]
    fn unknown_token_is_none_not_zero() {
        let mut vocab = Vocab::default();
        vocab.get_or_insert(b"known");
        assert_eq!(vocab.get(b"unknown"), None);
        assert_eq!(vocab.token(999), None);
        // No reserved UNK: id 0 is a real token.
        assert_eq!(vocab.token(0), Some(&b"known"[..]));
    }

    #[test]
    fn arbitrary_bytes_are_valid_tokens() {
        let mut vocab = Vocab::default();
        // Not valid UTF-8: partial multi-byte sequence.
        let invalid: &[u8] = &[0xF0, 0x9F];
        let id = vocab.get_or_insert(invalid);
        assert_eq!(vocab.get(invalid), Some(id));
        assert_eq!(vocab.token(id), Some(invalid));
    }

    #[test]
    fn save_load_roundtrip_preserves_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.bin");

        let mut vocab = Vocab::default();
        for token in [
            b"alpha".as_slice(),
            b"beta",
            &b"\n\n"[..],
            &[0xF0, 0x9F][..],
        ] {
            vocab.get_or_insert(token);
        }
        vocab.save(&path).unwrap();

        let loaded = Vocab::load(&path).unwrap();
        assert_eq!(loaded.len(), 4);
        for id in 0..4 {
            assert_eq!(loaded.token(id), vocab.token(id));
            assert_eq!(loaded.get(vocab.token(id).unwrap()), Some(id));
        }
    }

    #[test]
    fn whitespace_and_newline_tokens_survive_persistence() {
        // The reason persistence is binary: whitespace is token content.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.bin");

        let mut vocab = Vocab::default();
        let id_space = vocab.get_or_insert(b" ");
        let id_newlines = vocab.get_or_insert(b"\r\n\n");

        vocab.save(&path).unwrap();
        let loaded = Vocab::load(&path).unwrap();
        assert_eq!(loaded.token(id_space), Some(&b" "[..]));
        assert_eq!(loaded.token(id_newlines), Some(&b"\r\n\n"[..]));
    }

    #[test]
    fn corrupt_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.bin");
        std::fs::write(&path, b"definitely not bincode").unwrap();
        assert!(matches!(Vocab::load(&path), Err(VocabError::Format(_))));
    }

    #[test]
    fn wrong_version_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.bin");
        let file = VocabFile {
            version: 999,
            tokens: vec![b"x".to_vec()],
        };
        std::fs::write(&path, bincode::serialize(&file).unwrap()).unwrap();
        assert!(matches!(Vocab::load(&path), Err(VocabError::Format(_))));
    }

    #[test]
    fn duplicate_tokens_in_file_are_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.bin");
        let file = VocabFile {
            version: FORMAT_VERSION,
            tokens: vec![b"x".to_vec(), b"x".to_vec()],
        };
        std::fs::write(&path, bincode::serialize(&file).unwrap()).unwrap();
        assert!(matches!(Vocab::load(&path), Err(VocabError::Format(_))));
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            Vocab::load(&dir.path().join("nope.bin")),
            Err(VocabError::Io(_))
        ));
    }
}
