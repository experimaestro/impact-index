use std::fmt;

use crate::base::{DocId, ImpactValue};
use serde::{Deserialize, Serialize};

pub mod builder;
pub mod compress;
pub mod index;
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
