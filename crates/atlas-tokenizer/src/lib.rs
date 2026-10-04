//! Byte-level BPE tokenizer for Atlas.
//!
//! This crate is a hand-rolled implementation of byte pair encoding, written
//! to understand the mechanism rather than to wrap an existing one. It
//! matches tiktoken's cl100k_base token-for-token, which is verified by a
//! diff test against `tiktoken-rs`.
//!
//! The pieces:
//!
//! * [`ranks`] — merge-rank tables and the tiktoken `.tiktoken` file format.
//! * [`pretokenize`] — GPT-style pre-tokenization, which decides where merges
//!   may never cross.
//! * [`bpe`] — the two merge engines (naive rescan and priority queue with a
//!   linked list) used by encoding.
//! * [`encoder`] — [`encoder::BpeTokenizer`], the user-facing encode/decode
//!   type built from a rank table.
//! * [`trainer`] — BPE training over a corpus.

pub mod bpe;
pub mod encoder;
pub mod pretokenize;
pub mod ranks;
pub mod trainer;

pub use encoder::BpeTokenizer;
pub use ranks::{Rank, Ranks};
pub use trainer::train;
