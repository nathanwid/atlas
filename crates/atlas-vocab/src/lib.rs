//! Token ID management and persistence for Atlas (M03).
//!
//! A [`Vocab`] maps tokenizer output — byte-level BPE tokens, which are raw
//! byte strings — to stable, dense numeric IDs, and persists that mapping.
//!
//! Ownership boundary: `atlas-tokenizer` owns BPE mechanics, merge ranks,
//! encoding, and decoding; this crate owns vocabulary entries, IDs, lookup,
//! and persistence only. It never touches merge ranks.
//!
//! Design notes:
//!
//! * Keys are **raw token bytes** (`Box<[u8]>`). Byte-level tokens can be
//!   invalid UTF-8 (partial multi-byte sequences), so a string key would
//!   need escaping or lossy conversion; bytes preserve the tokenizer's
//!   semantics exactly.
//! * IDs are **dense** (`id ==` index into the token list), assigned by
//!   first occurrence during [`build`].
//! * Unknown tokens are handled explicitly with `Option`; there is no
//!   reserved UNK token.
//! * Persistence is a versioned bincode file. Binary is required, not a
//!   style choice: tokens contain whitespace and newlines, so a line-based
//!   text format would be ambiguous.

pub mod builder;
pub mod vocab;

pub use builder::build;
pub use vocab::{TokenId, Vocab, VocabError};
