//! Vocabulary management for mapping between string terms and term indices.
//!
//! Provides [`Vocabulary`] for bidirectional term <-> [`TermIndex`] mapping,
//! with CBOR-based serialization.

pub mod analyzer;
pub mod stemmer;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::base::TermIndex;

/// Bidirectional mapping between string terms and [`TermIndex`].
#[derive(Serialize, Deserialize, Clone)]
pub struct Vocabulary {
    term_to_id: HashMap<String, TermIndex>,
    id_to_term: Vec<String>,
}

impl Vocabulary {
    /// Create an empty vocabulary.
    pub fn new() -> Self {
        Self {
            term_to_id: HashMap::new(),
            id_to_term: Vec::new(),
        }
    }

    /// Get the index for a term, inserting it if not present.
    pub fn get_or_insert(&mut self, term: &str) -> TermIndex {
        if let Some(&id) = self.term_to_id.get(term) {
            id
        } else {
            let id = self.id_to_term.len();
            self.id_to_term.push(term.to_string());
            self.term_to_id.insert(term.to_string(), id);
            id
        }
    }

    /// Lookup a term's index (returns None if not present).
    pub fn get(&self, term: &str) -> Option<TermIndex> {
        self.term_to_id.get(term).copied()
    }

    /// Reverse lookup: get the term string for an index.
    pub fn term(&self, id: TermIndex) -> &str {
        &self.id_to_term[id]
    }

    /// Number of terms in the vocabulary.
    pub fn len(&self) -> usize {
        self.id_to_term.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_term.is_empty()
    }

    /// Save vocabulary to a CBOR file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let file = File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        ciborium::ser::into_writer(self, file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    /// Load vocabulary from a CBOR file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        ciborium::de::from_reader(file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}
