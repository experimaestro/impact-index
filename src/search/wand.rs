//! Implementation for WAND and Block-Max WAND algorithms

use std::collections::HashMap;

use log::debug;

use crate::{
    base::{DocId, ImpactValue},
    search::{ScoredDocument, TopScoredDocuments},
};

use crate::base::TermIndex;

use crate::index::{BlockTermImpactIterator, SparseIndex};

/**
 * WAND algorithm
 *
 *  Broder, A. Z., Carmel, D., Herscovici, M., Soffer, A. & Zien, J.
 * Efficient query evaluation using a two-level retrieval process.
 * Proceedings of the twelfth international conference on Information and knowledge management 426–434
 * (Association for Computing Machinery, 2003).
 * DOI 10.1145/956863.956944.
*/

/// Wraps an iterator with a query weight
struct BlockTermImpactIteratorWrapper<'a> {
    iterator: Box<dyn BlockTermImpactIterator + 'a>,
    query_weight: f32,
}

impl std::fmt::Display for BlockTermImpactIteratorWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "({}; max={})",
            self.iterator.current(),
            self.iterator.max_value() * self.query_weight
        )
    }
}

struct WandSearch<'a> {
    cur_doc: Option<DocId>,
    iterators: Vec<BlockTermImpactIteratorWrapper<'a>>,
}

impl<'a> WandSearch<'a> {
    fn new<'b: 'a>(index: &'b dyn SparseIndex, query: &HashMap<TermIndex, ImpactValue>) -> Self {
        let mut iterators = Vec::new();

        for (&ix, &weight) in query.iter() {
            // Discard a term if the index does not match
            if ix >= index.len() {
                debug!("Discarding term with index {}", ix);
                continue;
            }

            let iterator = index.block_iterator(ix);

            let mut wrapper = BlockTermImpactIteratorWrapper {
                iterator: iterator,
                query_weight: weight,
            };
            if wrapper.iterator.next_min_doc_id(0).is_some() {
                iterators.push(wrapper)
            }
        }

        Self {
            cur_doc: None,
            iterators: iterators,
        }
    }

    fn find_pivot_term(&mut self, theta: f32) -> Option<usize> {
        // Sort iterators by increasing document ID
        self.iterators
            .sort_by(|a, b| a.iterator.current().docid.cmp(&b.iterator.current().docid));

        // Accumulate until we get a value greater than theta
        let mut upper_bound = 0.;
        for (ix, iterator) in self.iterators.iter().enumerate() {
            upper_bound += iterator.iterator.max_value() * iterator.query_weight;
            if upper_bound > theta {
                return Some(ix);
            }
        }

        None
    }

    fn pick_term(&self, _up_to: usize) -> usize {
        // We just pick the first term (might not be the wisest)
        0
    }

    fn advance(&mut self, ix: usize, pivot: DocId) {
        let term_ix = self.pick_term(ix);
        if self.iterators[term_ix]
            .iterator
            .next_min_doc_id(pivot)
            .is_none()
        {
            // Remove this iterator
            self.iterators.remove(term_ix);
        }
    }

    fn next(&mut self, theta: ImpactValue) -> Option<DocId> {
        loop {
            if let Some(ix) = self.find_pivot_term(theta) {
                // Pivot term has been found
                let pivot = self.iterators[ix].iterator.current().docid;

                if match self.cur_doc {
                    Some(cur) => pivot <= cur,
                    None => false,
                } {
                    // Pivot has already been considered, advance one iterator
                    debug!(
                        "Pivot {} has already been considered [{}], advancing",
                        pivot, ix
                    );
                    self.advance(ix, pivot);
                } else if self.iterators[0].iterator.current().docid == pivot {
                    /* Success: all preceding terms belong to the pivot */
                    self.cur_doc = Some(pivot);
                    debug!("Computing score of {}", pivot);
                    return self.cur_doc;
                } else {
                    /* not enough mass */
                    self.advance(ix, pivot);
                }
            } else {
                return None;
            }
        }
    }
}

/**
 * Search using the WAND algorithmw
 */
pub fn search_wand<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let mut search = WandSearch::new(index, query);

    let mut results = TopScoredDocuments::new(top_k);
    let mut theta: ImpactValue = 0.;

    // Loop until there are no more candidates
    while let Some(candidate) = search.next(theta) {
        // Compute the score of the candidate
        let mut score: ImpactValue = 0.;
        for x in search.iterators.iter() {
            let c = x.iterator.current();
            if c.docid != candidate {
                break;
            }
            score += x.query_weight * c.value;
        }

        // Update the heap
        theta = results.add(candidate, score).max(0.);
    }

    results.into_sorted_vec()
}
