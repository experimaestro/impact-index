use std::{collections::{HashMap, BinaryHeap}, cmp::{Ordering}};

use crate::search::{ScoredDocument, DocId};

use super::{TermImpact, TermIndex};


/**
 * WAND algorithm
 * 
 *  Broder, A. Z., Carmel, D., Herscovici, M., Soffer, A. & Zien, J. Efficient query evaluation using a two-level retrieval process. in Proceedings of the twelfth international conference on Information and knowledge management 426–434 (Association for Computing Machinery, 2003). doi:10.1145/956863.956944.
*/

pub trait WandIterator<'a> {
    fn next(&mut self, doc_id: DocId) -> bool;
    
    /// Returns the current term impact
    fn current(&self) -> &TermImpact;

    /// Returns the term maximum impact
    fn max(&self) -> f64;
}

pub trait WandIndex<'a> {
    fn iterator(&'a self, term_ix: TermIndex) -> Box<dyn WandIterator<'a> + 'a>;
}

struct WandIteratorWrapper<'a> {
    iterator: Box::<dyn WandIterator<'a> + 'a>,
    query_weight: f64
}

impl PartialEq for ScoredDocument {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredDocument {}


impl PartialOrd for ScoredDocument {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.docid.partial_cmp(&other.docid) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for ScoredDocument {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want the minimum score
        if self.score == other.score {
            return Ordering::Equal;
        }
        if self.score < other.score {
            return Ordering::Less;
        }
        Ordering::Greater
    }
}

struct WandSearch<'a> {
    cur_doc: DocId,
    iterators: Vec<WandIteratorWrapper<'a>>
}

impl<'a> WandSearch<'a> {
    fn new(index: &'a dyn WandIndex<'a>, query: &HashMap<TermIndex, f64>) -> Self {
        let mut iterators = Vec::new();
        for mut iterator in query.into_iter().map(|(&ix, &weight)| WandIteratorWrapper {
            iterator: index.iterator(ix),
            query_weight: weight
        }) {
            if iterator.iterator.next(0) {
                iterators.push(iterator)
            }
        }

        Self {
            cur_doc: -1,
            iterators: iterators
        }
    }

    fn find_pivot_term(&mut self, theta: f64) -> Option<usize> {
        self.iterators.sort_by(|a, b| a.iterator.current().docid.cmp(&b.iterator.current().docid));

        let mut upper_bound = 0.;
        for (ix, iterator) in self.iterators.iter().enumerate() {
            // let &ti  = iterator.current();
            upper_bound += iterator.iterator.max() * iterator.query_weight;
            if upper_bound > theta {
                return Some(ix);
            }
        }
        None
    }

    fn pick_term(&self, up_to: usize) -> usize {
        0
    }

    fn next(&mut self, theta: f64) -> Option<DocId> {
        println!("Searching with theta {}", theta);

        loop {
            if let Some(ix) = self.find_pivot_term(theta) {
                let pivot = self.iterators[ix].iterator.current().docid;
                
                if pivot <= self.cur_doc {
                    let term_ix = self.pick_term(ix);
                    if !self.iterators[term_ix].iterator.next(pivot) {
                        // Remove this iterator
                        self.iterators.remove(ix);
                    }
                } else if self.iterators[0].iterator.current().docid == pivot {
                    /* Success: all preceding terms belong to the pivot */
                    self.cur_doc = pivot;
                    return Some(pivot);
                } else {
                    /* not enough mass */
                    let term_ix = self.pick_term(ix);
                    if !self.iterators[term_ix].iterator.next(pivot) {
                        // Remove this iterator
                        self.iterators.remove(ix);
                    }
                }
            }
        } 

    }
}

/**
 * Search using the WAND algorithmw
 */
pub fn search_wand<'a>(index: &'a dyn WandIndex<'a>, query: &HashMap<TermIndex, f64>, top_k: usize) -> Vec<ScoredDocument> {
    let mut search : WandSearch<'a> = WandSearch::new(index, query);

    let mut results = BinaryHeap::<ScoredDocument>::new();
    let mut theta: f64 = 0.;

    while let Some(candidate) = search.next(theta) {
        println!("Got a new candidate {}", candidate);

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
        if results.len() < top_k {
            results.push(ScoredDocument { docid: candidate, score: score });
        } else if results.peek().expect("should not happen").score < score {
            results.pop();
            results.push(ScoredDocument { docid: candidate, score: score });
            theta = f64::max(score, theta);
        }

    }
    results.into_sorted_vec()
}