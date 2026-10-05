//! Brute-force exact search over contiguous storage.
//!
//! Vectors live in one row-major `Vec<f32>`: a vector's `id` is its row
//! number, and a scan is `n` dot products over one flat buffer — no
//! per-vector heap allocation, best possible locality. Vectors are
//! L2-normalized on insert and the query is normalized on search, so a dot
//! product *is* the cosine similarity, in [-1, 1].
//!
//! This is the honest baseline: exact top-k, O(n·d) per query. Top-k uses a
//! full sort of the hits; at Atlas's current scale (hundreds of chunks) the
//! sort is noise next to the scan, and clarity beats a selection heap. If
//! scan speed ever matters, it becomes part of whichever milestone needs it.

/// L2-normalize in place. A zero vector is left as-is rather than turned
/// into NaNs; scores against it will be 0.
fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v {
            *x /= norm;
        }
    }
}

/// Dot product of equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Debug)]
pub struct BruteForceIndex {
    dim: usize,
    /// Row-major: row `id` occupies `vectors[id * dim .. (id + 1) * dim]`.
    vectors: Vec<f32>,
}

impl BruteForceIndex {
    /// Build from a batch of vectors. All must have length `dim` and be
    /// finite; each is normalized. Ids are assigned in input order.
    pub fn build(dim: usize, vectors: &[Vec<f32>]) -> anyhow::Result<Self> {
        let mut index = Self::with_capacity(dim, vectors.len());
        for vector in vectors {
            index.add(vector)?;
        }
        Ok(index)
    }

    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        BruteForceIndex {
            dim,
            vectors: Vec::with_capacity(capacity * dim),
        }
    }

    /// Add one vector; returns its id.
    pub fn add(&mut self, vector: &[f32]) -> anyhow::Result<u32> {
        anyhow::ensure!(
            vector.len() == self.dim,
            "vector has dimension {}, index expects {}",
            vector.len(),
            self.dim
        );
        anyhow::ensure!(
            vector.iter().all(|x| x.is_finite()),
            "vector contains non-finite components"
        );
        let id = (self.vectors.len() / self.dim) as u32;
        let mut row = vector.to_vec();
        normalize(&mut row);
        self.vectors.extend_from_slice(&row);
        Ok(id)
    }

    /// Cosine similarity of `query` against every stored vector, best last
    /// (ascending). Exposed for tests and the evaluation harness.
    pub fn scores(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(
            query.len(),
            self.dim,
            "query dimension {} does not match index dimension {}",
            query.len(),
            self.dim
        );
        let mut q = query.to_vec();
        normalize(&mut q);
        self.vectors
            .chunks(self.dim)
            .map(|row| dot(&q, row))
            .collect()
    }
}

impl super::index::VectorIndex for BruteForceIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<super::index::SearchHit> {
        let mut hits: Vec<super::index::SearchHit> = self
            .scores(query)
            .into_iter()
            .enumerate()
            .map(|(id, score)| super::index::SearchHit {
                id: id as u32,
                score,
            })
            .collect();
        // Descending by score; ids ascending among exact ties so results
        // are deterministic.
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        hits.truncate(k);
        hits
    }

    fn len(&self) -> usize {
        self.vectors.len() / self.dim
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::VectorIndex;

    #[test]
    fn identical_vectors_score_one() {
        let index = BruteForceIndex::build(3, &[vec![2.0, 0.0, 0.0]]).unwrap();
        let hits = index.search(&[5.0, 0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 0);
        assert!(
            (hits[0].score - 1.0).abs() < 1e-6,
            "cosine of parallel vectors must be 1"
        );
    }

    #[test]
    fn ranks_exactly_by_cosine() {
        // Rows: [1,0], [0,1], [0.6,0.8]. Query [1,0]: cosines 1, 0, 0.6.
        let index =
            BruteForceIndex::build(2, &[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.6, 0.8]]).unwrap();
        let hits = index.search(&[1.0, 0.0], 3);
        let order: Vec<u32> = hits.iter().map(|h| h.id).collect();
        assert_eq!(order, vec![0, 2, 1]);
        assert!(
            (hits[1].score - 0.6).abs() < 1e-6,
            "id 2 is [0.6,0.8], cosine 0.6"
        );
    }

    #[test]
    fn k_is_clamped_to_vector_count() {
        let index = BruteForceIndex::build(2, &[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        assert_eq!(index.search(&[1.0, 0.0], 100).len(), 2);
        assert!(index.search(&[1.0, 0.0], 0).is_empty());
    }

    #[test]
    fn empty_index_returns_no_hits() {
        let index = BruteForceIndex::with_capacity(4, 0);
        assert!(index.is_empty());
        assert!(index.search(&[0.1, 0.2, 0.3, 0.4], 5).is_empty());
    }

    #[test]
    fn rejects_wrong_dim_and_non_finite() {
        let mut index = BruteForceIndex::with_capacity(3, 2);
        assert!(index.add(&[1.0, 2.0]).is_err());
        assert!(index.add(&[1.0, 2.0, f32::NAN]).is_err());
        assert!(index.add(&[1.0, 2.0, 3.0]).is_ok());
    }

    #[test]
    #[should_panic(expected = "query dimension")]
    fn search_with_wrong_query_dim_panics() {
        let index = BruteForceIndex::build(2, &[vec![1.0, 0.0]]).unwrap();
        let _ = index.search(&[1.0, 2.0, 3.0], 1);
    }

    #[test]
    fn matches_naive_cosine_recomputation() {
        // Deterministic pseudo-random vectors (LCG, no rand dependency):
        // every index score must equal a directly computed cosine within
        // float tolerance, over a few hundred vectors.
        let dim = 16;
        let mut state = 0x853c49e6748fea9bu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let vectors: Vec<Vec<f32>> = (0..300)
            .map(|_| {
                (0..dim)
                    .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
                    .collect()
            })
            .collect();
        let index = BruteForceIndex::build(dim, &vectors).unwrap();

        for trial in 0..20 {
            let query: Vec<f32> = (0..dim)
                .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let hits = index.search(&query, 10);
            assert_eq!(hits.len(), 10);
            for hit in &hits {
                let q_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let v_norm = vectors[hit.id as usize]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                let expected = dot(&query, &vectors[hit.id as usize]) / (q_norm * v_norm);
                assert!(
                    (hit.score - expected).abs() < 1e-4,
                    "trial {trial}: id {} score {} vs cosine {expected}",
                    hit.id,
                    hit.score
                );
            }
            // Scores must be non-increasing.
            for pair in hits.windows(2) {
                assert!(pair[0].score >= pair[1].score);
            }
        }
    }
}
