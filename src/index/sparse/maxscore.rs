//! MaxScore algorithm

use std::collections::{HashMap, HashSet};

use log::debug;

use crate::{
    base::{DocId, ImpactValue},
    search::{ScoredDocument, TopScoredDocuments},
};

use crate::base::TermIndex;

use super::{
    index::{BlockTermImpactIndex, BlockTermImpactIterator},
    TermImpact,
};

struct MaxScoreTermIterator<'a> {
    iterator: Box<dyn BlockTermImpactIterator + 'a>,
    term_index: usize,
    query_weight: f32,
    max_value: f64,

    // Impact with value query weight taken into account
    impact: TermImpact
}

impl MaxScoreTermIterator<'_> {
    /// Call iterator's next
    fn next(&mut self) -> bool {
        if let Some(mut impact) = self.iterator.next() {
            impact.value *= self.query_weight;
            self.impact = impact;
            true
        } else {
            false
        }
    }


    fn seek_gek(&'_ mut self, doc_id: DocId) -> Option<&'_ TermImpact> {
        debug!("[term {}] Searching for doc id >= {}", self.term_index, doc_id);
        if doc_id <= self.impact.docid {
            return Some(&self.impact);
        }

        if !self.iterator.next_min_doc_id(doc_id) {
            return None
        }
        let mut impact = self.iterator.current();
        impact.value *= self.query_weight;
        self.impact = impact;
        debug!("[term {}] Current impact is {} / {}", self.term_index, self.impact, doc_id);
        Some(&self.impact)
    }
}

/*
 * Search using the MaxScore algorithm
 * (algorithm 1 in Accelerating Learned Sparse Indexes Via Term Impact Decomposition, Mackenzie et al., 2022)
 */
pub fn search_maxscore<'a>(
    index: &'a dyn BlockTermImpactIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    // --- Initialize the structures

    let mut results = TopScoredDocuments::new(top_k);
    let mut iterators = Vec::new();
    let mut theta: f64;

    for (&ix, &weight) in query.iter() {
        let iterator = index.iterator(ix);

        let max_value = ((&iterator).max_value() * weight) as f64;

        let mut wrapper = MaxScoreTermIterator {
            iterator: iterator,
            query_weight: weight,
            term_index: ix,
            impact: TermImpact {
                value: 0.,
                docid: 0,
            },
            max_value: max_value,
        };

        if wrapper.next() {
            iterators.push(wrapper);
        }
    }

    // Sort iterators
    iterators.sort_by(|a, b| b.iterator.max_value().total_cmp(&a.iterator.max_value()));

    let mut active = HashSet::<usize>::new();
    for i in 0..iterators.len() {
        active.insert(i);
    }

    let mut passive = HashSet::<usize>::new();
    let mut sum_pass = 0.;

    while !&active.is_empty() {
        // select next document, match all cursors
        let candidate: DocId = (&active).iter().fold(DocId::MAX as DocId, |cur, t| {
            cur.min(iterators[*t].impact.docid)
        });

        // score document
        let mut score = 0f64;
        passive.retain(|t| 
            if let Some(impact) = iterators[*t].seek_gek(candidate) {
                if candidate == impact.docid {
                    score += impact.value as f64;
                }
                true
            } else {
                false
            }
        );

        active.retain(|t| {
            if iterators[*t].impact.docid == candidate {
                score += iterators[*t].impact.value as f64;
                if !iterators[*t].next() {
                    return false;
                }
            }

            true
        });

        // check against heap, update if needed
        theta = results.add(candidate, score as f32).max(0.) as f64;

        // try to expand passive set
        let maybe_y = active.iter().reduce(|i, j| i.min(j)).and_then(|x| Some(*x));

        if let Some(y) = maybe_y {
            if iterators[y].max_value + sum_pass < theta {
                active.remove(&y);
                passive.insert(y);
                sum_pass += iterators[y].max_value;
            }
        }
    }

    results.into_sorted_vec()
}
