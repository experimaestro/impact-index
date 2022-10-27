use std::{collections::{HashMap}, f32::NEG_INFINITY};

use log::debug;

use crate::{search::{ScoredDocument, TopScoredDocuments}, base::DocId};

use crate::base::{TermIndex};

use super::TermImpact;

/**
 * WAND algorithm
 * 
 *  Broder, A. Z., Carmel, D., Herscovici, M., Soffer, A. & Zien, J. 
 * Efficient query evaluation using a two-level retrieval process.
 * Proceedings of the twelfth international conference on Information and knowledge management 426–434 
 * (Association for Computing Machinery, 2003). 
 * DOI 10.1145/956863.956944.
*/

pub trait WandIterator {
    /// Moves to the next document whose id is greater or equal than doc_id
    fn next(&mut self, doc_id: DocId) -> bool;
    
    /// Returns the current term impact
    fn current(&self) -> &TermImpact;

    /// Returns the term maximum impact
    fn max(&self) -> f64;
}

pub trait WandIndex {
    /// Returns a WAND iterator for a given term
    /// 
    /// ## Arguments
    /// 
    /// * `term_ix` The index of the term
    fn iterator<'a>(&'a self, term_ix: TermIndex) -> Box<dyn WandIterator + 'a>;
}

/// Wraps an iterator with a query weight
struct WandIteratorWrapper<'a> {
    iterator: Box<dyn WandIterator + 'a>,
    query_weight: f64
}

impl std::fmt::Display for WandIteratorWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}; max={})", self.iterator.current(), self.iterator.max() * self.query_weight)
    }
}

struct WandSearch<'a> {
    cur_doc: Option<DocId>,
    iterators: Vec<WandIteratorWrapper<'a>>
}

impl<'a> WandSearch<'a> {
    fn new<'b: 'a>(index: &'b dyn WandIndex, query: &HashMap<TermIndex, f64>) -> Self {
        let mut iterators = Vec::new();

        for (&ix, &weight) in query.iter() {
            let iterator = index.iterator(ix);

            let mut wrapper = WandIteratorWrapper {
                iterator: iterator,
                query_weight: weight
            };
            if wrapper.iterator.next(0) {
                iterators.push(wrapper)
            }
        }

        Self {
            cur_doc: None,
            iterators: iterators
        }
    }

    fn find_pivot_term(&mut self, theta: f64) -> Option<usize> {
        // Sort iterators by increasing document ID
        self.iterators.sort_by(|a, b| a.iterator.current().docid.cmp(&b.iterator.current().docid));
        
        // Accumulate until we get a value greater than theta
        let mut upper_bound = 0.;
        for (ix, iterator) in self.iterators.iter().enumerate() {
            upper_bound += iterator.iterator.max() * iterator.query_weight;
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

    fn next(&mut self, theta: f64) -> Option<DocId> {
        loop {
            if let Some(ix) = self.find_pivot_term(theta) {
                // Pivot term has been found
                let pivot = self.iterators[ix].iterator.current().docid;
                
                if match self.cur_doc {
                    Some(cur) => pivot <= cur,
                    None => false
                } {
                    // Pivot has already been considered, advance one iterator
                    debug!("Pivot {} has already been considered [{}], advancing", pivot, ix);
                    let term_ix = self.pick_term(ix);
                    if !self.iterators[term_ix].iterator.next(pivot) {
                        // Remove this iterator
                        self.iterators.remove(term_ix);
                    }
                } else if self.iterators[0].iterator.current().docid == pivot {
                    /* Success: all preceding terms belong to the pivot */
                    self.cur_doc = Some(pivot);
                    debug!("Computing score of {}", pivot);
                    return self.cur_doc;
                } else {
                    /* not enough mass */
                    let term_ix = self.pick_term(ix);
                    if !self.iterators[term_ix].iterator.next(pivot) {
                        // Remove this iterator
                        self.iterators.remove(term_ix);
                    }
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
pub fn search_wand<'a>(index: &'a mut dyn WandIndex, query: &HashMap<TermIndex, f64>, top_k: usize) -> Vec<ScoredDocument> {
    let mut search = WandSearch::new(index, query);

    let mut results = TopScoredDocuments::new(top_k);
    let mut theta: f64 = 0.;

    // Loop until there are no more candidates
    while let Some(candidate) = search.next(theta) {
        // Compute the score of the candidate
        let mut score: f64 = 0.;
        for x in search.iterators.iter() {
            let c = x.iterator.current();
            if c.docid != candidate {
                break;
            }
            score +=  x.query_weight * c.value as f64;
        }

        // Update the heap
        theta = results.add(candidate, score).max(0.);
        
    }

    results.into_sorted_vec()
}