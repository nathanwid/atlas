//! Merge-rank tables and the tiktoken `.tiktoken` file format.
//!
//! A BPE vocabulary is a map from byte strings to ranks. The rank is the
//! order in which the token was created during training; encoding replays
//! merges in rank order, so "lowest rank wins" is the whole algorithm.

use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Read};

/// The numeric identity of a token. tiktoken uses `u32` ranks too, so this
/// matches the diff-test oracle without conversions.
pub type Rank = u32;

/// A BPE merge table: token bytes -> rank.
pub type Ranks = HashMap<Box<[u8]>, Rank>;

/// Errors from [`parse_tiktoken_ranks`].
#[derive(Debug)]
pub enum RanksError {
    Io(std::io::Error),
    /// A line was not `<base64> <rank>`.
    MalformedLine(usize),
    /// The base64 payload of a line did not decode.
    BadBase64(usize),
    /// The rank field of a line was not a valid `u32`.
    BadRank(usize),
    /// The same byte string appeared with two different ranks.
    DuplicateToken(usize),
}

impl fmt::Display for RanksError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RanksError::Io(e) => write!(f, "failed to read ranks file: {e}"),
            RanksError::MalformedLine(n) => write!(f, "line {n}: expected `<base64> <rank>`"),
            RanksError::BadBase64(n) => write!(f, "line {n}: invalid base64 token"),
            RanksError::BadRank(n) => write!(f, "line {n}: invalid rank"),
            RanksError::DuplicateToken(n) => write!(f, "line {n}: duplicate token bytes"),
        }
    }
}

impl std::error::Error for RanksError {}

impl From<std::io::Error> for RanksError {
    fn from(e: std::io::Error) -> Self {
        RanksError::Io(e)
    }
}

/// Parse a tiktoken-format ranks file.
///
/// Each line is `<base64 token> <rank>`. The file format carries the token as
/// raw bytes (base64-encoded), not as text, which is what makes this a
/// byte-level vocabulary: tokens can contain arbitrary bytes, including
/// invalid UTF-8.
///
/// Base64 decoding is implemented here instead of pulling in the `base64`
/// crate; the format only needs the standard alphabet and it keeps the
/// production dependency set minimal.
pub fn parse_tiktoken_ranks(reader: impl Read) -> Result<Ranks, RanksError> {
    let reader = BufReader::new(reader);
    let mut ranks = Ranks::with_capacity(100_256);
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let lineno = i + 1;
        let Some((token_b64, rank_str)) = line.split_once(' ') else {
            return Err(RanksError::MalformedLine(lineno));
        };
        let token = base64_decode(token_b64).ok_or(RanksError::BadBase64(lineno))?;
        let rank: Rank = rank_str
            .trim()
            .parse()
            .map_err(|_| RanksError::BadRank(lineno))?;
        if ranks.insert(token.into(), rank).is_some() {
            return Err(RanksError::DuplicateToken(lineno));
        }
    }
    Ok(ranks)
}

/// Decode standard-alphabet base64 (`A-Z a-z 0-9 + /`, `=` padding).
///
/// Accumulates 6 bits per input character and flushes full bytes, which is
/// the whole algorithm; padding is honored only by being stripped, since the
/// leftover bits are already a whole number of bytes for valid input.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn value(b: u8) -> Option<u32> {
        match b {
            b'A'..=b'Z' => Some((b - b'A') as u32),
            b'a'..=b'z' => Some((b - b'a') as u32 + 26),
            b'0'..=b'9' => Some((b - b'0') as u32 + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let s = s.as_bytes();
    let stripped = match s.iter().position(|&b| b == b'=') {
        Some(i) => &s[..i],
        None => s,
    };
    let mut out = Vec::with_capacity(stripped.len() * 3 / 4 + 1);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in stripped {
        buf = (buf << 6) | value(b)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    // Canonical input has no stray bits left over after the last full byte.
    if bits > 0 && buf & ((1 << bits) - 1) != 0 {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decodes_known_values() {
        assert_eq!(base64_decode("IQ==").unwrap(), vec![b'!']);
        assert_eq!(base64_decode("Ig==").unwrap(), vec![b'"']);
        assert_eq!(base64_decode("YQ==").unwrap(), vec![b'a']);
        assert_eq!(base64_decode("YWI=").unwrap(), b"ab".to_vec());
        assert_eq!(base64_decode("YWJj").unwrap(), b"abc".to_vec());
        assert_eq!(base64_decode("SGVsbG8=").unwrap(), b"Hello".to_vec());
        assert!(base64_decode("IQ!").is_none());
    }

    #[test]
    fn parses_two_line_table() {
        let input = "IQ== 0\nIg== 1\n";
        let ranks = parse_tiktoken_ranks(input.as_bytes()).unwrap();
        assert_eq!(ranks.len(), 2);
        assert_eq!(ranks[&b"!"[..]], 0);
        assert_eq!(ranks[&b"\""[..]], 1);
    }

    #[test]
    fn rejects_malformed_lines() {
        assert!(matches!(
            parse_tiktoken_ranks("no-space-here\n".as_bytes()),
            Err(RanksError::MalformedLine(1))
        ));
        assert!(matches!(
            parse_tiktoken_ranks("!# 0\n".as_bytes()),
            Err(RanksError::BadBase64(1))
        ));
    }
}
