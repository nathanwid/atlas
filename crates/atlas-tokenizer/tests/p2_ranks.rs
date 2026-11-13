use atlas_tokenizer::{BpeTokenizer, ranks::Ranks};
fn bytes() -> Ranks {
    (0..256u32)
        .map(|i| (vec![i as u8].into_boxed_slice(), i))
        .collect()
}
#[test]
fn invalid_complete_tables_are_rejected_at_construction() {
    for kind in 0..3 {
        let mut ranks = bytes();
        match kind {
            0 => {
                ranks.insert(Box::from(&b"ab"[..]), 0);
            }
            1 => {
                ranks.remove(&b"a"[..]);
            }
            _ => {
                ranks.insert(Box::from(&b""[..]), 300);
            }
        };
        assert!(
            std::panic::catch_unwind(|| BpeTokenizer::from_ranks(ranks)).is_err(),
            "kind {kind}"
        );
    }
}
#[test]
fn non_contiguous_unique_ranks_roundtrip() {
    let ranks = bytes().into_iter().map(|(t, r)| (t, r * 2)).collect();
    let tok = BpeTokenizer::from_ranks(ranks);
    let text = "caf\u{e9} \u{65e5}\u{672c}\u{8a9e}";
    assert_eq!(tok.decode(&tok.encode(text)).unwrap(), text);
}
