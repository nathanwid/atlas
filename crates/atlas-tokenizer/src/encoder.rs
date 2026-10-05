//! The user-facing tokenizer: encode text to token ranks, decode ranks back.
//!
//! A [`BpeTokenizer`] is just a rank table plus the pre-tokenizer and one of
//! the merge engines. Decoding needs the inverse table, which is cheap to
//! keep alongside; byte-level vocabularies are total, so every byte of any
//! input can always be encoded and every token can always be decoded.

use crate::pretokenize::pretokenize;
use crate::ranks::{Rank, Ranks};
use std::collections::HashMap;
use std::fmt;

/// Errors from [`BpeTokenizer::decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The rank is not in the vocabulary.
    UnknownRank(Rank),
    /// The decoded bytes were not valid UTF-8. Only possible for tables that
    /// were not built by this crate's trainer or a tiktoken-format file.
    InvalidUtf8,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnknownRank(r) => write!(f, "unknown token rank {r}"),
            DecodeError::InvalidUtf8 => write!(f, "decoded bytes are not valid UTF-8"),
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    ranks: Ranks,
    reverse: HashMap<Rank, Box<[u8]>>,
}

impl BpeTokenizer {
    /// Build a tokenizer from a merge table.
    ///
    /// The inverse table is built eagerly. For cl100k that is 100k small
    /// entries, microseconds of work, and it keeps `decode` allocation-free
    /// beyond the output buffer.
    pub fn from_ranks(ranks: Ranks) -> Self {
        let mut reverse: HashMap<Rank, Box<[u8]>> = HashMap::with_capacity(ranks.len());
        for (token, &rank) in &ranks {
            reverse.insert(rank, token.clone());
        }
        BpeTokenizer { ranks, reverse }
    }

    /// Load a tokenizer from a tiktoken-format ranks file
    /// (`<base64 token> <rank>` per line, e.g. `cl100k_base.tiktoken`).
    pub fn from_tiktoken_reader(
        reader: impl std::io::Read,
    ) -> Result<Self, crate::ranks::RanksError> {
        Ok(Self::from_ranks(crate::ranks::parse_tiktoken_ranks(
            reader,
        )?))
    }

    pub fn encode(&self, text: &str) -> Vec<Rank> {
        self.encode_with(text, crate::bpe::encode_piece_heap)
    }

    /// Encode with the naive rescan engine. Exists for the M1 benchmark
    /// comparing merge strategies; both engines must agree, which the tests
    /// verify.
    pub fn encode_naive(&self, text: &str) -> Vec<Rank> {
        self.encode_with(text, crate::bpe::encode_piece_naive)
    }

    fn encode_with(&self, text: &str, engine: crate::bpe::PieceEncoder) -> Vec<Rank> {
        // English text lands near 4 tokens per 3 bytes with cl100k; the
        // capacity is a cheap guess that avoids most regrowth.
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        for piece in pretokenize(text) {
            out.extend(engine(piece.as_bytes(), &self.ranks));
        }
        out
    }

    /// Decode token ranks back to the exact original bytes.
    ///
    /// Byte-level vocabularies roundtrip perfectly: encoding never throws
    /// away information, so decode(encode(x)) == x for all x.
    pub fn decode(&self, tokens: &[Rank]) -> Result<String, DecodeError> {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for &rank in tokens {
            let token = self
                .reverse
                .get(&rank)
                .ok_or(DecodeError::UnknownRank(rank))?;
            bytes.extend_from_slice(token);
        }
        String::from_utf8(bytes).map_err(|_| DecodeError::InvalidUtf8)
    }

    pub fn ranks(&self) -> &Ranks {
        &self.ranks
    }

    /// Byte length of one token. Chunking needs this to split an oversized
    /// pre-token at token boundaries without decoding: byte-level tokens
    /// partition their input bytes exactly, so byte lengths are enough to
    /// slice the original text.
    pub fn token_byte_len(&self, rank: Rank) -> Option<usize> {
        self.reverse.get(&rank).map(|token| token.len())
    }

    /// The bytes of one token. Vocabulary building needs the reverse of
    /// `encode`: ranks come out, bytes go in the vocabulary. Byte-level
    /// tokens can be arbitrary byte strings, so this returns bytes, not a
    /// `String`.
    pub fn token_bytes(&self, rank: Rank) -> Option<&[u8]> {
        self.reverse.get(&rank).map(|token| &token[..])
    }

    pub fn vocab_size(&self) -> usize {
        self.ranks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> BpeTokenizer {
        // Byte-level vocabularies are total: every possible byte must have a
        // rank, or encoding arbitrary text cannot work. The toy table mirrors
        // that with all 256 single-byte tokens plus one merge.
        let mut ranks = Ranks::new();
        for b in 0u8..=255 {
            ranks.insert(vec![b].into(), b as Rank);
        }
        ranks.insert(b"ab".to_vec().into(), 256);
        BpeTokenizer::from_ranks(ranks)
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tok = toy();
        let text = "abba cabba abc";
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids).unwrap(), text);
    }

    #[test]
    fn decode_rejects_unknown_rank() {
        let tok = toy();
        // 0..=255 are all single-byte ranks in the toy table; 999 is not.
        assert_eq!(tok.decode(&[999]), Err(DecodeError::UnknownRank(999)));
    }
}
