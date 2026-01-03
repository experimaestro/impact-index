//! Bounded heap for tracking top-K scores efficiently.
//!
//! Used to compute k-th percentile scores (10th, 100th, 1000th) without
//! storing all scores in memory.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A bounded min-heap that keeps track of the top-K largest scores.
///
/// This allows computing k-th percentile scores with O(K) memory instead
/// of O(N) memory required by sorting all scores.
pub struct TopKHeap {
    /// Min-heap storing top scores (using Reverse for min-heap behavior)
    heap: BinaryHeap<Reverse<u8>>,
    /// Maximum size of the heap
    max_size: usize,
}

impl TopKHeap {
    /// Maximum K we need to track (for 1000th percentile)
    pub const MAX_K: usize = 1000;

    /// Creates a new TopKHeap with the default max size (1000).
    pub fn new() -> Self {
        Self::with_capacity(Self::MAX_K)
    }

    /// Creates a new TopKHeap with a specific capacity.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(max_size),
            max_size,
        }
    }

    /// Pushes a score into the heap.
    ///
    /// If the heap is at capacity, only keeps the score if it's larger
    /// than the current minimum.
    #[inline]
    pub fn push(&mut self, score: u8) {
        if self.heap.len() < self.max_size {
            self.heap.push(Reverse(score));
        } else if let Some(&Reverse(min)) = self.heap.peek() {
            if score > min {
                self.heap.pop();
                self.heap.push(Reverse(score));
            }
        }
    }

    /// Returns the k-th percentile scores as [s10th, s100th, s1000th].
    ///
    /// These correspond to the scores at positions 9, 99, and 999 (0-indexed)
    /// when scores are sorted in descending order.
    pub fn get_kth_scores(&self) -> Vec<u8> {
        let mut sorted: Vec<u8> = self.heap.iter().map(|r| r.0).collect();
        sorted.sort_by(|a, b| b.cmp(a)); // Descending order

        vec![
            sorted.get(9).copied().unwrap_or(0),   // 10th highest
            sorted.get(99).copied().unwrap_or(0),  // 100th highest
            sorted.get(999).copied().unwrap_or(0), // 1000th highest
        ]
    }

    /// Returns the number of scores currently in the heap.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns true if the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Clears the heap for reuse.
    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

impl Default for TopKHeap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_heap() {
        let heap = TopKHeap::new();
        let scores = heap.get_kth_scores();
        assert_eq!(scores, vec![0, 0, 0]);
    }

    #[test]
    fn test_fewer_than_10_scores() {
        let mut heap = TopKHeap::new();
        for i in 1..=5 {
            heap.push(i);
        }
        let scores = heap.get_kth_scores();
        assert_eq!(scores, vec![0, 0, 0]); // Not enough scores for 10th
    }

    #[test]
    fn test_exactly_10_scores() {
        let mut heap = TopKHeap::new();
        for i in 1..=10 {
            heap.push(i);
        }
        let scores = heap.get_kth_scores();
        assert_eq!(scores[0], 1); // 10th highest is 1
        assert_eq!(scores[1], 0); // Not enough for 100th
        assert_eq!(scores[2], 0); // Not enough for 1000th
    }

    #[test]
    fn test_100_scores() {
        let mut heap = TopKHeap::new();
        for i in 1..=100 {
            heap.push(i as u8);
        }
        let scores = heap.get_kth_scores();
        assert_eq!(scores[0], 91); // 10th highest: 100, 99, 98, 97, 96, 95, 94, 93, 92, 91
        assert_eq!(scores[1], 1); // 100th highest
        assert_eq!(scores[2], 0); // Not enough for 1000th
    }

    #[test]
    fn test_bounded_size() {
        let mut heap = TopKHeap::new();
        // Push 2000 scores
        for i in 0..2000 {
            heap.push((i % 256) as u8);
        }
        // Heap should only contain 1000 elements
        assert_eq!(heap.len(), 1000);
    }

    #[test]
    fn test_keeps_largest() {
        let mut heap = TopKHeap::with_capacity(5);
        for i in [1, 5, 3, 8, 2, 9, 4, 7, 6, 10] {
            heap.push(i);
        }
        // Should keep [10, 9, 8, 7, 6]
        let mut sorted: Vec<u8> = heap.heap.iter().map(|r| r.0).collect();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(sorted, vec![10, 9, 8, 7, 6]);
    }
}
