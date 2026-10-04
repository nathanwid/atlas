//! Shared test/bench helpers: fixture loading, oracle wiring, and a
//! HuggingFace `tokenizers` tokenizer derived from the same cl100k ranks.
//!
//! Included by integration tests via `mod common;` and by the benchmark via
//! `#[path]`, so the exact same fixture loading and HF construction run in
//! both places. Each including binary uses a different subset, hence the
//! module-wide dead_code allowance.

#![allow(dead_code)]

use atlas_tokenizer::{BpeTokenizer, Ranks};
use std::path::PathBuf;

pub fn fixture_path(name: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name]
        .iter()
        .collect()
}

/// The committed cl100k_base ranks (100,256 tokens, OpenAI's published file).
pub fn load_ranks() -> Ranks {
    let f = std::fs::File::open(fixture_path("cl100k.tiktoken")).expect("ranks fixture");
    atlas_tokenizer::ranks::parse_tiktoken_ranks(std::io::BufReader::new(f))
        .expect("ranks fixture must parse")
}

pub fn load_atlas_bpe() -> BpeTokenizer {
    BpeTokenizer::from_ranks(load_ranks())
}

pub fn load_corpus() -> String {
    std::fs::read_to_string(fixture_path("corpus.txt")).expect("corpus fixture")
}

/// Texts chosen to stress every alternative of the pre-tokenization pattern
/// and every byte range: contractions, digit runs, punctuation, unicode,
/// emoji, CRLF, tabs, trailing whitespace.
pub fn sample_texts() -> Vec<&'static str> {
    vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "numbers 1234567890 and 3.14159 and 007",
        "I'll don't we've they're I'M you'd it's",
        "   leading and   trailing spaces   ",
        "line1\r\nline2\n\nline3\n",
        "tabs\tand\tspaces  mixed",
        "na\u{ef}ve caf\u{e9} — \u{65e5}\u{672c}\u{8a9e} \u{1F600} \u{1F980}",
        "punctuation!!! ??? ... ---([{<=>}])",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "snake_case and kebab-case and CamelCase123",
        "",
        " ",
        "\n",
        "Atlas is a small search engine that learns to read.",
    ]
}

/// The GPT-2 byte-to-unicode alphabet used by HuggingFace tokenizers.
///
/// HF's BPE works on unicode strings, not bytes, so each byte is mapped to a
/// visible char (printable ASCII to itself, everything else to chars 256+).
/// The mapping is a bijection, which is why HF byte-level tokenizers also
/// roundtrip exactly. Our tokenizer works on raw bytes; to compare against
/// HF we translate our byte tokens into this alphabet.
pub fn byte_to_unicode() -> Vec<char> {
    let printable =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut table = vec!['\0'; 256];
    let mut n = 0u32;
    for b in 0u32..256 {
        table[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

/// Reconstruct HF-format merges from tiktoken ranks.
///
/// tiktoken's rank table is the training order: a token of rank r was formed
/// by merging two tokens of lower rank. HF's BPE orders merges by list
/// position, so every merge that produces token T is emitted at position
/// rank(T).
///
/// Every valid split of T must be emitted, not just one: HF merges greedily
/// and may build T's two halves through a different grouping than tiktoken's
/// training did (e.g. it can reach "llo" instead of "Hell" inside "Hello"),
/// and a missing split would strand that path partway through a word.
pub fn derive_merges(ranks: &Ranks) -> Vec<(String, String)> {
    let max_rank = ranks.values().copied().max().expect("non-empty ranks") as usize;
    let mut by_rank: Vec<Option<&[u8]>> = vec![None; max_rank + 1];
    for (token, &rank) in ranks {
        by_rank[rank as usize] = Some(token);
    }
    let table = byte_to_unicode();
    let to_hf = |bytes: &[u8]| -> String { bytes.iter().map(|&b| table[b as usize]).collect() };

    let mut merges = Vec::new();
    for token in by_rank.into_iter().flatten() {
        if token.len() < 2 {
            continue;
        }
        for i in 1..token.len() {
            if ranks.contains_key(&token[..i]) && ranks.contains_key(&token[i..]) {
                merges.push((to_hf(&token[..i]), to_hf(&token[i..])));
            }
        }
    }
    merges
}

/// Build a HuggingFace `tokenizers` tokenizer equivalent to cl100k_base,
/// derived from the same committed ranks file the Atlas tokenizer uses.
///
/// Pre-tokenization is a Split with the cl100k pattern (Isolated = matches
/// become their own pieces, i.e. find-all semantics) followed by ByteLevel
/// with `use_regex(false)`, which only performs the byte-to-unicode mapping
/// without applying GPT-2's own splitting pattern.
pub fn build_hf_cl100k(ranks: &Ranks) -> tokenizers::Tokenizer {
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::tokenizer::SplitDelimiterBehavior;

    let table = byte_to_unicode();
    let vocab: Vocab = ranks
        .iter()
        .map(|(bytes, &rank)| {
            let s: String = bytes.iter().map(|&b| table[b as usize]).collect();
            (s, rank)
        })
        .collect();
    let merges = derive_merges(ranks);
    let bpe = BPE::new(vocab, merges);

    let split = Split::new(
        SplitPattern::Regex(atlas_tokenizer::pretokenize::CL100K_PATTERN.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .expect("cl100k pattern must compile under onig");
    let byte_level = ByteLevel::new(false, false, false);
    let sequence = Sequence::new(vec![split.into(), byte_level.into()]);

    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(sequence));
    tokenizer
}
