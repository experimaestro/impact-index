use std::fmt;

use ndarray::{ArrayBase, Ix1, Data, Dim};

pub type DocId = u64;
pub type TermIndex = usize;
pub type ImpactValue = f32;

/// Impact value and its document
struct TermImpact {
    docid: DocId,
    value: ImpactValue
}

impl std::fmt::Display for TermImpact {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({},{})", self.docid, self.value)
    }
}

/// The set of postings
struct TermsImpacts {
    postings: Vec<Vec<TermImpact>>
}

impl TermsImpacts {
    /// Create a new term impacts in memory structure 
    fn new(nterms: u32) -> TermsImpacts {
        let mut impacts = TermsImpacts { postings: Vec::new() };

        for i in 1..nterms {
            impacts.postings.push(Vec::new())
        };

        impacts
    
    }

    fn add_impact(&mut self, termix: TermIndex, docid: DocId, value: ImpactValue) {
        self.postings[termix].push(TermImpact {
            docid: docid,
            value: value
        })
    }
}


/// The indexation
pub struct Indexer {
    impacts: TermsImpacts
}

impl Indexer {
    pub fn new(nterms: u32) -> Indexer {
        Indexer { impacts: TermsImpacts::new(nterms) }
    }

    pub fn add<S, T>(&mut self, docid: DocId, terms: &ArrayBase<S, Ix1>, values: &ArrayBase<T, Ix1>) 
    where S: Data<Elem = TermIndex>, T: Data<Elem = ImpactValue>
    {
        assert!(terms.len() == values.len(), "Value and term lists should have the same length");
        for ix in 1..terms.len() {
            self.impacts.add_impact(terms[ix], docid, values[ix]) 
        }
    }
}