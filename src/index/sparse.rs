use std::fmt;

use ndarray::{ArrayBase, Ix1, Data};

pub type DocId = u64;
pub type TermIndex = usize;
pub type ImpactValue = f32;

/** Contains the impacts */
struct TermImpact {
    docid: DocId,
    value: ImpactValue
}

impl std::fmt::Display for TermImpact {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({},{})", self.docid, self.value)
    }
}

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

pub fn test() {
    let mut impacts = TermsImpacts::new(232);

    impacts.add_impact(3, 932, 1.23);

    println!("{}", impacts.postings[3][0]);

}

/// The indexation
pub struct Indexer {
    impacts: TermsImpacts
}

impl Indexer {
    pub fn new() -> Indexer {
        Indexer { impacts: TermsImpacts::new(100) }
    }

    pub fn add<S, T>(&mut self, docid: DocId, terms: &ArrayBase<S, Ix1>, values: &ArrayBase<T, Ix1>) 
    where S: Data<Elem = TermIndex>, T: Data<Elem = ImpactValue>
    {
        for ix in 1..terms.len() {
            &self.impacts.add_impact(terms[ix], docid, values[ix]) 
        }
    }
}