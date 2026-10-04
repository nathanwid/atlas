//! GPT-style pre-tokenization.
//!
//! Pre-tokenization runs before any merging. It cuts the text into pieces and
//! guarantees that BPE merges never cross a piece boundary. Without it, a
//! merge could glue the end of one word onto the start of the next and the
//! vocabulary would fill with junk.
//!
//! The cl100k pattern, one alternative per kind of piece:
//!
//! * `'(?i:[sdmt]|ll|ve|re)` — contractions (clitic suffixes) stay attached
//!   to nothing before them; they are their own piece, case-insensitively.
//! * `[^\r\n\p{L}\p{N}]?+\p{L}++` — a letter run, optionally preceded by one
//!   non-letter/non-digit character (which is how a single leading space
//!   sticks to its word).
//! * `\p{N}{1,3}+` — digits in groups of at most three ("1234567" splits as
//!   123 / 456 / 7), so numbers of any length tokenize compositionally.
//! * ` ?[^\s\p{L}\p{N}]++[\r\n]*` — punctuation runs, again with an optional
//!   leading space, plus any directly trailing newlines.
//! * `\s++$` — trailing whitespace at end of input.
//! * `\s*[\r\n]` — whitespace ending in one newline (keeps blank-line runs
//!   as single pieces ending exactly at the newline).
//! * `\s+(?!\S)` — whitespace not followed by a non-space, i.e. all but the
//!   last space of a run before a word.
//! * `\s` — any remaining single whitespace.
//!
//! The pattern is copied verbatim from `tiktoken-rs` 0.12.1 so the diff test
//! compares tokenizer logic, not pattern drift. It needs lookahead and
//! possessive quantifiers, which the `regex` crate does not support, hence
//! `fancy-regex` (the same engine tiktoken-rs uses).

use fancy_regex::Regex;
use std::sync::OnceLock;

/// The cl100k_base pre-tokenization pattern, verbatim from tiktoken-rs.
pub const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s+";

fn cl100k_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(CL100K_PATTERN).expect("cl100k pre-tokenization pattern must compile")
    })
}

/// Split text into pre-token pieces.
///
/// The pieces are borrows of `text` and their concatenation reproduces
/// `text` exactly; the cl100k alternatives cover every possible character.
/// If that coverage invariant were ever broken by a pattern change, encoding
/// would silently drop bytes, so a gap is treated as a bug and panics rather
/// than losing data. The roundtrip tests prove the invariant holds.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let re = cl100k_regex();
    let mut pieces = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        let m = m.expect("pattern has no backreferences and cannot fail mid-iteration");
        debug_assert!(m.start() >= last, "pre-tokenization pieces overlap");
        if m.start() != last {
            panic!(
                "pre-tokenization gap: bytes {last}..{} are unmatched",
                m.start()
            );
        }
        last = m.end();
        pieces.push(m.as_str());
    }
    if last != text.len() {
        panic!(
            "pre-tokenization gap: trailing bytes {last}..{} are unmatched",
            text.len()
        );
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_reconstruct_input() {
        let samples = [
            "Hello, world!",
            "  leading and   trailing spaces   ",
            "line1\r\nline2\n\nline3\n",
            "numbers 1234567890 and 3.14159",
            "I'll don't we've they're I'M",
            "tab\tseparated\tvalues",
            "no_ascii_\u{1F600}_here",
        ];
        for s in samples {
            assert_eq!(pretokenize(s).concat(), s);
        }
    }

    #[test]
    fn digits_group_in_threes() {
        assert_eq!(pretokenize("1234567"), vec!["123", "456", "7"]);
    }

    #[test]
    fn contraction_is_own_piece() {
        assert_eq!(pretokenize("don't"), vec!["don", "'t"]);
    }
}
