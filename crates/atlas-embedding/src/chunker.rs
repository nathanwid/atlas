//! Document chunking by token count.
//!
//! A long document makes a weak single embedding: a vector that must stand
//! for everything stands for nothing. Chunking bounds what each vector has
//! to represent. The size is measured in tokens, not characters, because the
//! embedder consumes tokens.
//!
//! Chunks are cut only at pre-token boundaries (never mid-word) using the
//! same pre-tokenizer the tokenizer itself uses. The pieces are guaranteed
//! to reconstruct the document exactly, so with `overlap_tokens == 0` the
//! chunks concatenate back to the original text byte-for-byte.
//!
//! One pre-token can exceed `max_tokens` (e.g. a very long punctuation or
//! letter run, since the pattern has no length cap for those). Such a piece
//! is hard-split at token boundaries using the tokenizer's byte-level
//! guarantee: tokens partition their piece's bytes exactly, so summing
//! [`BpeTokenizer::token_byte_len`] gives exact split offsets without
//! decoding. Overlap is not carried across a hard-split piece.

use atlas_tokenizer::BpeTokenizer;
use atlas_tokenizer::pretokenize::pretokenize;

/// One chunk: a borrow of the document plus its starting byte offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk<'a> {
    pub text: &'a str,
    pub start: usize,
}

/// Splits documents into token-bounded chunks at pre-token boundaries.
#[derive(Debug)]
pub struct Chunker<'t> {
    tokenizer: &'t BpeTokenizer,
    max_tokens: usize,
    overlap_tokens: usize,
}

impl<'t> Chunker<'t> {
    /// `overlap_tokens` carries trailing pieces into the next chunk so
    /// adjacent chunks share context. The carry is at piece granularity, so
    /// the shared region is the largest piece-aligned suffix whose token
    /// count fits in `overlap_tokens`.
    pub fn new(tokenizer: &'t BpeTokenizer, max_tokens: usize, overlap_tokens: usize) -> Self {
        assert!(max_tokens >= 1, "chunks must hold at least one token");
        assert!(
            overlap_tokens < max_tokens,
            "overlap must be smaller than the chunk size or chunks cannot advance"
        );
        Chunker {
            tokenizer,
            max_tokens,
            overlap_tokens,
        }
    }

    /// Chunk `document`. With zero overlap, concatenating `chunk.text` in
    /// order reproduces `document` exactly. Deterministic by construction:
    /// no hashing, no parallelism, no HashMap iteration.
    pub fn chunk<'a>(&self, document: &'a str) -> Vec<Chunk<'a>> {
        let pieces = pretokenize(document);
        if pieces.is_empty() {
            return Vec::new();
        }
        // Byte offset of every piece; pieces cover the document exactly.
        let mut offsets = Vec::with_capacity(pieces.len() + 1);
        let mut pos = 0;
        for piece in &pieces {
            offsets.push(pos);
            pos += piece.len();
        }
        offsets.push(pos);

        // Token count per piece, computed once. English documents produce
        // short pieces, so this is a handful of small encodes per document.
        let piece_tokens: Vec<usize> = pieces
            .iter()
            .map(|p| self.tokenizer.encode(p).len())
            .collect();

        let mut chunks = Vec::new();
        let mut start = 0;
        while start < pieces.len() {
            // Greedy fill: take whole pieces while they fit.
            let mut count = 0;
            let mut end = start;
            while end < pieces.len() && count + piece_tokens[end] <= self.max_tokens {
                count += piece_tokens[end];
                end += 1;
            }

            if end == start {
                // A single pre-token longer than max_tokens: hard-split it.
                // Byte-level tokens partition the piece's bytes exactly
                // (hence exact offsets), but a token boundary can fall
                // inside a multi-byte character — cl100k, for example,
                // encodes a 4-byte emoji as a 3-byte token plus a 1-byte
                // token. Cuts therefore happen at token boundaries that are
                // also char boundaries; when the max-token cut point is
                // mid-character, the window absorbs tokens until the next
                // boundary, so such windows may exceed max_tokens. This can
                // only occur inside pieces whose tokens split characters.
                let piece = pieces[start];
                let base = offsets[start];
                let tokens = self.tokenizer.encode(piece);
                let mut window_start = 0;
                let mut offset = 0;
                let mut window_tokens = 0;
                for &rank in &tokens {
                    let len = self
                        .tokenizer
                        .token_byte_len(rank)
                        .expect("encode only emits ranks from this vocabulary");
                    if window_tokens > 0
                        && piece.is_char_boundary(offset)
                        && window_tokens + 1 > self.max_tokens
                    {
                        chunks.push(Chunk {
                            text: &piece[window_start..offset],
                            start: base + window_start,
                        });
                        window_start = offset;
                        window_tokens = 0;
                    }
                    offset += len;
                    window_tokens += 1;
                }
                chunks.push(Chunk {
                    text: &piece[window_start..offset],
                    start: base + window_start,
                });
                start += 1;
                continue;
            }

            chunks.push(Chunk {
                text: &document[offsets[start]..offsets[end]],
                start: offsets[start],
            });

            // Carry piece-aligned overlap into the next chunk. The guard
            // `j > start + 1` keeps at least one piece of forward progress.
            if self.overlap_tokens > 0 {
                let mut j = end;
                let mut carried = 0;
                while j > start + 1 && carried + piece_tokens[j - 1] <= self.overlap_tokens {
                    j -= 1;
                    carried += piece_tokens[j];
                }
                start = j;
            } else {
                start = end;
            }
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> BpeTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../atlas-tokenizer/tests/fixtures/cl100k.tiktoken"
        );
        let f = std::fs::File::open(path).expect("cl100k ranks fixture");
        BpeTokenizer::from_tiktoken_reader(std::io::BufReader::new(f)).expect("parses")
    }

    fn corpus() -> String {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../atlas-tokenizer/tests/fixtures/corpus.txt"
        );
        std::fs::read_to_string(path).expect("corpus fixture")
    }

    #[test]
    fn chunks_reconstruct_document_without_overlap() {
        let tok = tokenizer();
        let corpus = corpus();
        let chunker = Chunker::new(&tok, 64, 0);
        let chunks = chunker.chunk(&corpus);
        assert!(chunks.len() > 1, "corpus must actually split at 64 tokens");
        let joined: String = chunks.iter().map(|c| c.text).collect();
        assert_eq!(joined, corpus);
        // Offsets are exact and increasing.
        let mut expected = 0;
        for c in &chunks {
            assert_eq!(c.start, expected);
            expected += c.text.len();
        }
    }

    #[test]
    fn chunks_respect_max_tokens() {
        let tok = tokenizer();
        let corpus = corpus();
        for max in [8, 64, 256] {
            let chunker = Chunker::new(&tok, max, 0);
            for c in chunker.chunk(&corpus) {
                assert!(
                    tok.encode(c.text).len() <= max,
                    "chunk at {} has {} tokens > {max}",
                    c.start,
                    tok.encode(c.text).len()
                );
            }
        }
    }

    #[test]
    fn chunking_is_deterministic() {
        let tok = tokenizer();
        let corpus = corpus();
        let a = Chunker::new(&tok, 32, 8).chunk(&corpus);
        let b = Chunker::new(&tok, 32, 8).chunk(&corpus);
        assert_eq!(a, b);
    }

    #[test]
    fn overlap_shares_byte_identical_region() {
        let tok = tokenizer();
        let doc = "The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs. \
                   How vexingly quick daft zebras jump!";
        let chunks = Chunker::new(&tok, 12, 4).chunk(doc);
        assert!(chunks.len() > 1);
        for pair in chunks.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            // The next chunk must begin inside the previous chunk and the
            // shared bytes must be identical.
            let a_tail = a.start + a.text.len();
            assert!(
                b.start > a.start && b.start < a_tail,
                "no overlap between chunks"
            );
            assert_eq!(
                &a.text[b.start - a.start..],
                &b.text[..a_tail - b.start],
                "overlapping regions differ"
            );
        }
    }

    #[test]
    fn oversized_pretoken_hard_splits_at_token_boundaries() {
        let tok = tokenizer();
        // A 1000-byte punctuation run is one pre-token; 8 tokens cannot hold it.
        let doc = format!("ok {} end", "!".repeat(1000));
        let chunker = Chunker::new(&tok, 8, 0);
        let chunks = chunker.chunk(&doc);
        assert!(chunks.len() > 10);
        let joined: String = chunks.iter().map(|c| c.text).collect();
        assert_eq!(joined, doc);
        for c in &chunks {
            assert!(tok.encode(c.text).len() <= 8);
        }
    }

    #[test]
    fn empty_document_yields_no_chunks() {
        let tok = tokenizer();
        assert!(Chunker::new(&tok, 16, 0).chunk("").is_empty());
    }

    #[test]
    fn multibyte_characters_survive_hard_split() {
        let tok = tokenizer();
        // cl100k encodes each 4-byte emoji as a 3-byte token plus a 1-byte
        // token, so a long run is one pre-token whose token boundaries fall
        // mid-character. Cuts must land on char boundaries and chunks must
        // still reconstruct the document.
        let doc = "\u{1F600}".repeat(50);
        let chunker = Chunker::new(&tok, 4, 0);
        let chunks = chunker.chunk(&doc);
        assert!(chunks.len() > 5);
        let joined: String = chunks.iter().map(|c| c.text).collect();
        assert_eq!(joined, doc);
        for c in &chunks {
            assert!(tok.encode(c.text).len() <= 4);
        }
    }
}
