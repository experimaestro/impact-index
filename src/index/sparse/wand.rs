use std::{collections::{HashMap}};

use log::debug;

use crate::{search::{ScoredDocument, TopScoredDocuments}, base::{DocId, ImpactValue}};

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
    fn next_min_doc_id(&mut self, doc_id: DocId) -> bool;
    
    /// Returns the current term impact (only valid when the iterator is here)
    fn current(&self) -> &TermImpact;

    /// Returns the term maximum impact
    fn max_value(&self) -> ImpactValue;

    /// Returns the term maximum impact
    fn max_doc_id(&self) -> DocId;

    /// Returns the length
    fn length(&self) -> usize;

    /// Returns the next element
    fn next(&mut self) -> Option<TermImpact> {
        if self.next_min_doc_id(0) {
            Some(*self.current())
        } else {
            None
        }
    }
}


pub struct ValueIterator<'a> {
    iterator: Box<dyn WandIterator + 'a>
}

impl<'a> Iterator for ValueIterator<'a> {
    type Item = ImpactValue;

    fn next(&mut self) -> Option<ImpactValue> {
        if let Some(ti) = self.iterator.next() {
            Some(ti.value)
        } else {
            None
        }
    }

    fn max(self) -> Option<Self::Item> {
        return Some(self.iterator.max_value())
    }
}

struct DocIdIterator<'a> {
    iterator: Box<dyn WandIterator + 'a>
}
impl<'a> Iterator for DocIdIterator<'a> {
    type Item = DocId;

    fn next(&mut self) -> Option<DocId> {
        if let Some(ti) = self.iterator.next() {
            Some(ti.docid)
        } else {
            None
        }
    }

    fn max(self) -> Option<Self::Item> {
        return Some(self.iterator.max_doc_id())
    }
}

pub trait WandIndex {
    /// Returns a WAND iterator for a given term
    /// 
    /// ## Arguments
    /// 
    /// * `term_ix` The index of the term
    fn iterator(&self, term_ix: TermIndex) -> Box<dyn WandIterator + '_>;

    /// Returns the number of terms in the index
    fn length(&self) -> usize;

    fn values_iterator(&self, term_ix: TermIndex) -> ValueIterator<'_> {
        ValueIterator { iterator: self.iterator(term_ix) }
    }
}

/// Wraps an iterator with a query weight
struct WandIteratorWrapper<'a> {
    iterator: Box<dyn WandIterator + 'a>,
    query_weight: f32
}

impl std::fmt::Display for WandIteratorWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}; max={})", self.iterator.current(), self.iterator.max_value() * self.query_weight)
    }
}

struct WandSearch<'a> {
    cur_doc: Option<DocId>,
    iterators: Vec<WandIteratorWrapper<'a>>
}

impl<'a> WandSearch<'a> {
    fn new<'b: 'a>(index: &'b dyn WandIndex, query: &HashMap<TermIndex, ImpactValue>) -> Self {
        let mut iterators = Vec::new();

        for (&ix, &weight) in query.iter() {
            let iterator = index.iterator(ix);

            let mut wrapper = WandIteratorWrapper {
                iterator: iterator,
                query_weight: weight
            };
            if wrapper.iterator.next_min_doc_id(0) {
                iterators.push(wrapper)
            }
        }

        Self {
            cur_doc: None,
            iterators: iterators
        }
    }

    fn find_pivot_term(&mut self, theta: f32) -> Option<usize> {
        // Sort iterators by increasing document ID
        self.iterators.sort_by(|a, b| a.iterator.current().docid.cmp(&b.iterator.current().docid));
        
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
        if !self.iterators[term_ix].iterator.next_min_doc_id(pivot) {
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
                    None => false
                } {
                    // Pivot has already been considered, advance one iterator
                    debug!("Pivot {} has already been considered [{}], advancing", pivot, ix);
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
pub fn search_wand<'a>(index: &'a dyn WandIndex, query: &HashMap<TermIndex, ImpactValue>, top_k: usize) -> Vec<ScoredDocument> {
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
            score +=  x.query_weight * c.value;
        }

        // Update the heap
        theta = results.add(candidate, score).max(0.);
        
    }

    results.into_sorted_vec()
}