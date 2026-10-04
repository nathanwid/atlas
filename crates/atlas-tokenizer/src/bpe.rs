//! The two BPE merge engines used by encoding.
//!
//! Encoding one pre-token piece means: repeatedly find the adjacent pair
//! whose concatenation has the lowest rank in the merge table and merge it,
//! until no adjacent pair is in the table.
//!
//! Two engines implement that contract and must produce identical output:
//!
//! * [`encode_piece_naive`] rescans the whole piece for the best pair after
//!   every merge. O(n^2) pair lookups for a piece of n bytes, but trivially
//!   correct and a good baseline.
//! * [`encode_piece_heap`] keeps a priority queue of candidate pairs plus a
//!   doubly linked list of live pieces, so each merge costs O(log n) instead
//!   of a full rescan. Stale queue entries are detected and skipped lazily
//!   rather than removed eagerly, because a binary heap cannot delete
//!   arbitrary entries efficiently.
//!
//! Tie-breaking needs no coordination between the engines: the rank of a
//! pair is the rank of its concatenated bytes, so two different byte strings
//! never share a rank, and equal-rank pairs are the same bytes in different
//! places, whose merge order does not change the final sequence.

use crate::ranks::{Rank, Ranks};
use std::cmp::Reverse;

/// Signature shared by both merge engines.
pub type PieceEncoder = fn(&[u8], &Ranks) -> Vec<Rank>;

/// Heap entry: (merge rank, left node idx, left end, right end, right node
/// idx). The lengths recorded at push time are what makes stale entries
/// detectable on pop.
type MergeCandidate = Reverse<(Rank, usize, u32, u32, usize)>;

/// Encode one piece by rescanning for the lowest-ranked pair after each merge.
///
/// Parts are kept as `(start, end)` ranges into `piece`. Adjacent parts are
/// contiguous in the original bytes, so a pair lookup is a slice of `piece`
/// and no concatenation ever allocates.
pub fn encode_piece_naive(piece: &[u8], ranks: &Ranks) -> Vec<Rank> {
    if piece.is_empty() {
        return Vec::new();
    }
    let mut parts: Vec<(u32, u32)> = (0..piece.len() as u32).map(|i| (i, i + 1)).collect();
    loop {
        let mut best: Option<(Rank, usize)> = None;
        for i in 0..parts.len() - 1 {
            let (start, _) = parts[i];
            let (_, end) = parts[i + 1];
            let Some(&r) = ranks.get(&piece[start as usize..end as usize]) else {
                continue;
            };
            if best.is_none_or(|(best_rank, _)| r < best_rank) {
                best = Some((r, i));
            }
        }
        let Some((_, i)) = best else { break };
        parts[i].1 = parts[i + 1].1;
        parts.remove(i + 1);
    }
    parts
        .iter()
        .map(|&(s, e)| ranks[&piece[s as usize..e as usize]])
        .collect()
}

/// Encode one piece with a priority queue of pairs over a linked list of parts.
///
/// The linked list is index-based (`Vec<Node>`), not pointer-based: merging
/// extends the left node in place and kills the right node, so no allocation
/// happens per merge and no references between nodes exist.
///
/// A heap entry `(rank, left_idx, left_end, right_end, right_idx)` is a
/// promise that those two nodes were adjacent with those lengths at push
/// time. On pop the promise is re-checked; any mismatch means the entry is
/// stale (a neighbor merged since) and it is discarded. This lazy
/// invalidation is what makes each merge O(log n) amortized.
pub fn encode_piece_heap(piece: &[u8], ranks: &Ranks) -> Vec<Rank> {
    use std::collections::BinaryHeap;

    let n = piece.len();
    if n == 0 {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct Node {
        start: u32,
        end: u32,
        prev: i32,
        next: i32,
    }

    let mut nodes: Vec<Node> = (0..n)
        .map(|i| Node {
            start: i as u32,
            end: i as u32 + 1,
            prev: i as i32 - 1,
            next: i as i32 + 1,
        })
        .collect();
    nodes[n - 1].next = -1;
    // Nodes never move; the list only shrinks. Node 0 is always the head
    // because merges extend their left node in place.
    let mut alive = vec![true; n];

    let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::with_capacity(n);
    for i in 0..n - 1 {
        if let Some(&r) = ranks.get(&piece[i..i + 2]) {
            heap.push(Reverse((r, i, i as u32 + 1, i as u32 + 2, i + 1)));
        }
    }

    while let Some(Reverse((_, li, left_end, right_end, ri))) = heap.pop() {
        if !alive[li] || !alive[ri] {
            continue;
        }
        if nodes[li].end != left_end || nodes[ri].end != right_end {
            continue; // one side merged since this entry was pushed
        }
        if nodes[li].next != ri as i32 {
            continue; // no longer adjacent
        }
        // The promise holds: merge right into left.
        nodes[li].end = right_end;
        alive[ri] = false;
        let next = nodes[ri].next;
        nodes[li].next = next;
        if next >= 0 {
            nodes[next as usize].prev = li as i32;
        }
        // The two pairs that changed: (li, its new next) and (li's prev, li).
        if next >= 0 {
            let next_end = nodes[next as usize].end;
            if let Some(&r) = ranks.get(&piece[nodes[li].start as usize..next_end as usize]) {
                heap.push(Reverse((r, li, right_end, next_end, next as usize)));
            }
        }
        if nodes[li].prev >= 0 {
            let pi = nodes[li].prev as usize;
            let prev_end = nodes[pi].end;
            if let Some(&r) = ranks.get(&piece[nodes[pi].start as usize..right_end as usize]) {
                heap.push(Reverse((r, pi, prev_end, right_end, li)));
            }
        }
    }

    let mut out = Vec::with_capacity(n);
    let mut cur = 0usize;
    loop {
        let node = nodes[cur];
        out.push(ranks[&piece[node.start as usize..node.end as usize]]);
        if node.next < 0 {
            break;
        }
        cur = node.next as usize;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy merge table over the alphabet {a, b, c}: every byte is a token,
    /// plus a handful of merges. Single-byte ranks come first, matching how
    /// real BPE tables are laid out.
    fn toy_ranks() -> Ranks {
        let mut ranks = Ranks::new();
        for (i, b) in [b'a', b'b', b'c'].iter().enumerate() {
            ranks.insert(vec![*b].into(), i as Rank);
        }
        for (i, s) in [b"ab".as_slice(), b"bc", b"abc", b"aab", b"cc", b"abcc"]
            .iter()
            .enumerate()
        {
            ranks.insert(s.to_vec().into(), 3 + i as Rank);
        }
        ranks
    }

    /// Deterministic pseudo-random pieces from a tiny alphabet, so the two
    /// engines are compared over thousands of shapes without a rand
    /// dependency. An LCG is enough for shape diversity.
    #[test]
    fn engines_agree_on_random_pieces() {
        let ranks = toy_ranks();
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let len = (next() % 24) as usize + 1;
            let piece: Vec<u8> = (0..len).map(|_| b"abc"[(next() % 3) as usize]).collect();
            assert_eq!(
                encode_piece_naive(&piece, &ranks),
                encode_piece_heap(&piece, &ranks),
                "engines disagree on {piece:?}"
            );
        }
    }

    #[test]
    fn lowest_rank_wins_in_order() {
        let ranks = toy_ranks();
        // "abc": pair "bc" has rank 4, pair "ab" has rank 3, so "ab" merges
        // first, producing "abc" (rank 5) directly.
        assert_eq!(encode_piece_heap(b"abc", &ranks), vec![5]);
        // "aab": pair "aa" is not a token; pair "ab" (rank 3) merges, then
        // "aab" (rank 6) exists.
        assert_eq!(encode_piece_heap(b"aab", &ranks), vec![6]);
    }

    #[test]
    fn unknown_pairs_stay_unmerged() {
        let ranks = toy_ranks();
        // "ca" and "ac" are not tokens; nothing can merge.
        assert_eq!(encode_piece_heap(b"ca", &ranks), vec![2, 0]);
        assert_eq!(encode_piece_naive(b"ca", &ranks), vec![2, 0]);
    }
}
