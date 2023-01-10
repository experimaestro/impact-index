use std::collections::HashMap;
use std::fmt;

use crate::base::{DocId, ImpactValue, TermIndex};
use crate::search::ScoredDocument;
use index::BlockTermImpactIndex;
use serde::{Deserialize, Serialize};

pub mod builder;
pub mod compress;
pub mod index;
pub mod maxscore;
pub mod sharding;
pub mod wand;

/// Term impact = document ID + impact value
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct TermImpact {
    pub docid: DocId,
    pub value: ImpactValue,
}

impl std::fmt::Display for TermImpact {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({},{})", self.docid, self.value)
    }
}

/// An iterator on term impacts
pub type TermImpactIterator<'a> = Box<dyn Iterator<Item = TermImpact> + 'a + Send>;

/// A search function
pub type SearchFn = fn(
    index: &dyn BlockTermImpactIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument>;
