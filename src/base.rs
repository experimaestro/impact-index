//! Core type definitions and traits used throughout the library.

/// Index identifying a term in the vocabulary (0-based).
pub type TermIndex = usize;

/// Floating-point impact score assigned to a term in a document.
/// Must be strictly positive (> 0) when indexing.
pub type ImpactValue = f32;

/// Unique document identifier (monotonically increasing during indexing).
pub type DocId = u64;

/// Convenience alias for boxed error results.
pub type BoxResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Trait for objects that have a countable length (e.g., number of terms).
pub trait Len {
    /// Returns the number of elements.
    fn len(&self) -> usize;
}

use std::collections::HashMap;
use std::fmt;
use std::{fs::File, path::Path};

use crate::builder::load_forward_index;
use crate::index::{BlockTermImpactIterator, SparseIndex};
use crate::search::ScoredDocument;
use serde::{Deserialize, Serialize};

pub const INDEX_CBOR: &str = "index.cbor";
pub const BUILDER_INDEX_CBOR: &str = "information.cbor";

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
pub type TermImpactIterator<'a> = Box<dyn BlockTermImpactIterator + 'a>;

/// A search function
pub type SearchFn = fn(
    index: &dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument>;

/// Trait for deserializing and instantiating a [`SparseIndex`] from disk.
///
/// Implementations are stored alongside the index data (serialized via CBOR)
/// so the correct loader is automatically selected when loading.
#[typetag::serde(tag = "type")]
pub trait IndexLoader {
    /// Consumes the loader and returns a ready-to-query index.
    fn into_index(self: Box<Self>, path: &Path, in_memory: bool) -> Box<dyn SparseIndex>;
}

/// Loads a sparse index from the given directory.
///
/// Supports both legacy forward-index format (`information.cbor`) and
/// the newer loader-based format (`index.cbor`).
///
/// # Arguments
///
/// * `path` - Directory containing the index files
/// * `in_memory` - If `true`, loads data into memory; otherwise uses memory-mapped I/O
pub fn load_index(path: &Path, in_memory: bool) -> Box<dyn SparseIndex> {
    let info_path = path.join(BUILDER_INDEX_CBOR);
    if info_path.exists() {
        // Takes care of old format with the raw builder index
        Box::new(load_forward_index(path, in_memory))
    } else {
        // Uses the new way to load indices
        let info_path = path.join(INDEX_CBOR);
        let info_file = File::options()
            .read(true)
            .open(info_path)
            .expect("Error while opening the index information file");

        let loader: Box<dyn IndexLoader> = ciborium::de::from_reader(info_file)
            .expect("Error loading compressed term index information");

        loader.into_index(path, in_memory)
    }
}

/// Saves an index loader to disk so the index can be loaded later.
///
/// Serializes the loader as CBOR into `index.cbor` within the given directory.
pub fn save_index(loader: Box<dyn IndexLoader>, path: &Path) -> Result<(), std::io::Error> {
    let info_path = path.join(INDEX_CBOR);
    let info_path_s = info_path.display().to_string();

    let info_file = File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(info_path)
        .expect(&format!("Error while creating file {}", info_path_s));

    ciborium::ser::into_writer(&loader, info_file)
        .expect("Error saving compressed term index information");

    Ok(())
}
