use std::{collections::BinaryHeap, cmp::Ordering};

use crate::base::DocId;

pub struct ScoredDocument {
    pub docid: DocId,
    pub score: f64
}

impl PartialEq for ScoredDocument {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredDocument {}


impl PartialOrd for ScoredDocument {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // match self.docid.partial_cmp(&other.docid) {
        //     Some(core::cmp::Ordering::Equal) => {}
        //     ord => return ord,
        // }
        other.score.partial_cmp(&self.score)
    }
}

// Total order adds document ID
impl Ord for ScoredDocument {
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.total_cmp(&self.score)
        // // We want the minimum score
        // if self.score == other.score {
        //     return Ordering::Equal;
        // }
        // if self.score < other.score {
        //     return Ordering::Less;
        // }
        // Ordering::Greater
    }
}

pub struct TopScoredDocuments  {
    heap: BinaryHeap::<ScoredDocument>,
    top_k: usize
}

impl TopScoredDocuments {
    pub fn new(top_k: usize) -> Self {
        Self { heap: BinaryHeap::new(), top_k: top_k }
    }

    pub fn add(&mut self, candidate: DocId, score: f64) {
        if self.heap.len() < self.top_k {
            self.heap.push(ScoredDocument { docid: candidate, score: score });
        } else if self.heap.peek().expect("should not happen").score < score {
            self.heap.pop();
            self.heap.push(ScoredDocument { docid: candidate, score: score });
        }
    }

    pub fn into_sorted_vec(mut self) -> Vec<ScoredDocument> {
        self.heap.into_sorted_vec()
    }
}
