//! MaxScore algorithm

use std::collections::HashMap;


use crate::{
    base::{DocId, ImpactValue},
    search::{ScoredDocument, TopScoredDocuments},
};

use crate::base::TermIndex;

use super::{index::{BlockTermImpactIndex, BlockTermImpactIterator}};

struct MaxScoreTermIterator<'a> {
    iterator: Box<dyn BlockTermImpactIterator + 'a>,
    query_weight: f32,
    max_remaining: f32
}
struct MaxScoreSearch<'a> {
    cur_doc: Option<DocId>,
    iterators: Vec<MaxScoreTermIterator<'a>>,
}

// TODO: implement max score search with posting list clipping
// See "Accelerating Learned Sparse Indexes Via Term Impact Decomposition" (EMNLP 2022)
impl<'a> MaxScoreSearch<'a> {
    /// Initialize the search structure
    fn new<'b: 'a>(index: &'b dyn BlockTermImpactIndex, query: &HashMap<TermIndex, ImpactValue>) -> Self {
        let mut iterators = Vec::new();

        for (&ix, &weight) in query.iter() {
            let iterator = index.iterator(ix);

            let mut wrapper = MaxScoreTermIterator {
                iterator: iterator,
                query_weight: weight,
                max_remaining: 0f32
            };
            if wrapper.iterator.next_min_doc_id(0) {
                iterators.push(wrapper)
            }
        }

        // Sort iterators
        iterators.sort_by(|a, b| { 
            b.iterator.max_value().total_cmp(&a.iterator.max_value())
        });

        // Compute max
        let mut cum = 0f32;
        iterators.iter_mut().rev().for_each(|a| {
            a.max_remaining = cum;
            cum += a.iterator.max_value();
        });
        
        Self {
            cur_doc: None,
            iterators: iterators,
        }
    }


    fn next(&mut self, theta: ImpactValue) -> Option<DocId> {
        todo!("Not implemented");
    }
}

/**
 * Search using the WAND algorithmw
 */
pub fn search_maxscore<'a>(
    index: &'a dyn BlockTermImpactIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let mut search = MaxScoreSearch::new(index, query);

    let mut results = TopScoredDocuments::new(top_k);
    let mut theta: ImpactValue = 0.;

    // Loop until there are no more candidates
    while let Some(candidate) = search.next(theta) {
        // Compute the score of the candidate
        let mut score: ImpactValue = 0.;
        for x in search.iterators.iter() {
            let c = x.iterator.current();
            if c.docid == candidate {
                score += x.query_weight * c.value;
            }
        }

        // Update the heap
        theta = results.add(candidate, score).max(0.);
    }

    results.into_sorted_vec()
}
