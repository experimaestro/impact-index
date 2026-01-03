//! Mirror struct for BMP's Index, needed because BMP's Index has private fields.
//!
//! This struct must serialize identically to `bmp::index::inverted_index::Index`
//! using bincode to ensure file compatibility.

use bmp::index::posting_list::PostingList;
use fst::{Map, MapBuilder};
use serde::{Deserialize, Serialize};

/// Mirror of `bmp::index::inverted_index::Index` with public fields.
///
/// This allows us to construct an Index directly without going through
/// BMP's IndexBuilder, while maintaining serialization compatibility.
#[derive(Serialize, Deserialize)]
pub struct Index {
    num_documents: usize,
    posting_lists: Vec<PostingList>,
    #[serde(
        serialize_with = "serialize_fst_map",
        deserialize_with = "deserialize_fst_map"
    )]
    termmap: Map<Vec<u8>>,
    documents: Vec<String>,
}

impl Index {
    /// Creates a new Index with the given components.
    pub fn new(
        num_documents: usize,
        posting_lists: Vec<PostingList>,
        term_names: Vec<String>,
        documents: Vec<String>,
    ) -> Self {
        // Build FST term map
        // Terms must be inserted in lexicographical order for FST
        let mut indexed_terms: Vec<(usize, &String)> = term_names.iter().enumerate().collect();
        indexed_terms.sort_by(|a, b| a.1.cmp(b.1));

        let mut fst_builder = MapBuilder::memory();
        for (index, term) in indexed_terms {
            let _ = fst_builder.insert(term, index as u64);
        }
        let termmap = Map::new(fst_builder.into_inner().unwrap()).unwrap();

        Self {
            num_documents,
            posting_lists,
            termmap,
            documents,
        }
    }

    /// Returns the number of documents in the index.
    pub fn num_documents(&self) -> usize {
        self.num_documents
    }

    /// Returns a reference to the posting lists.
    pub fn posting_lists(&self) -> &[PostingList] {
        &self.posting_lists
    }

    /// Returns a reference to the document names.
    pub fn documents(&self) -> &[String] {
        &self.documents
    }
}

// Serialization function for the FST Map - matches BMP's implementation
fn serialize_fst_map<S>(termmap: &Map<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let bytes = termmap.as_fst().to_vec();
    serializer.serialize_bytes(&bytes)
}

// Deserialization function for the FST Map - matches BMP's implementation
fn deserialize_fst_map<'de, D>(deserializer: D) -> Result<Map<Vec<u8>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let bytes = Vec::<u8>::deserialize(deserializer)?;
    let map = Map::new(bytes).map_err(serde::de::Error::custom)?;
    Ok(map)
}
