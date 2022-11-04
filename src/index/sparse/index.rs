//! Main data structure used to describe an index

use serde::{Serialize, Deserialize};

use crate::base::{ImpactValue, DocId};

#[derive(Serialize, Deserialize)]
pub struct TermIndexPageInformation {
    /// Position for the document ID stream
    pub docid_position: u64,

    /// Position for the impact value stream
    pub value_position: u64,

    /// Number of records
    pub length: usize,

    /// Maximum value for this page
    pub max_value: ImpactValue,

    /// Maximum document ID for this page
    pub max_doc_id: DocId
}

impl TermIndexPageInformation {
    pub fn new() -> Self {
        Self {
            docid_position: 0,
            value_position: 0,
            length: 0,
            max_value: 0.,
            max_doc_id: 0
        }
    }
}

#[derive(Serialize, Deserialize)]
pub  struct TermIndexInformation {
    pub pages: Vec<TermIndexPageInformation>,
    pub max_value: ImpactValue,
    pub max_doc_id: DocId,
    pub length: usize
}

/// Global information on the index structure
#[derive(Serialize, Deserialize)]
pub struct IndexInformation {
    pub terms: Vec<TermIndexInformation>
}

impl IndexInformation {
    /// Creates a new index information
    pub fn new() -> IndexInformation{
        IndexInformation {
            terms: Vec::new()
        }
    }
}
