//! Methods for compressing the posting lists

use std::{
    cell::RefCell,
    fmt,
    fs::File,
    io::{Read, Seek, Write},
    path::Path,
};

use super::{
    index::{BlockTermImpactIndex, BlockTermImpactIterator},
    TermImpact,
};
use crate::{
    base::{DocId, ImpactValue, TermIndex},
    utils::buffer::{Buffer, MemoryBuffer, MmapBuffer, Slice},
};
use log::debug;
use serde::{Deserialize, Serialize};
use sucds::{EliasFanoBuilder, Searial};

//
// ---- Compression ---
//
pub trait Compressor<T> {
    fn write(&self, writer: &mut dyn Write, values: &[T]);
    fn read(&self, reader: &mut dyn Read) -> Box<dyn Iterator<Item = T>>;
}

#[typetag::serde(tag = "type")]
pub trait DocIdCompressor: Compressor<DocId> + Sync {}

#[typetag::serde(tag = "type")]
pub trait ValueCompressor: Compressor<ImpactValue> + Sync {}

//
// ---- Compressed index global information  ---
//

#[derive(Serialize, Deserialize)]
pub struct TermBlockInformation {
    /// Position within the document ID stream
    pub docid_position_range: (u64, u64),

    /// Position within the impact value stream
    pub value_position_range: (u64, u64),

    /// Number of records
    pub length: usize,

    /// Maximum value for this page
    pub max_value: ImpactValue,

    /// Maximum document ID for this page
    pub max_doc_id: DocId,
}

impl std::fmt::Display for TermBlockInformation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "(docids: {}-{}, impacts: {}-{}, len: {}, max_v: {}, max_docid: {})",
            self.docid_position_range.0,
            self.docid_position_range.1,
            self.value_position_range.0,
            self.value_position_range.1,
            self.length,
            self.max_value,
            self.max_doc_id
        )
    }
}

/// Block-based index information for a term
#[derive(Serialize, Deserialize)]
pub struct TermBlocksInformation {
    pub pages: Vec<TermBlockInformation>,
    pub max_value: ImpactValue,
    pub max_doc_id: DocId,
    pub length: usize,
}

/// Global information on the index structure
#[derive(Serialize, Deserialize)]
pub struct CompressedIndexInformation {
    pub terms: Vec<TermBlocksInformation>,
    doc_ids_compressor: Box<dyn DocIdCompressor>,
    values_compressor: Box<dyn ValueCompressor>,
}

pub struct CompressedIndex {
    information: CompressedIndexInformation,

    /// View on document IDs
    docid_buffer: Box<dyn Buffer>,

    /// View on document IDs
    impact_buffer: Box<dyn Buffer>,
}

//
// ---- Iterators over compressed block indices

pub struct CompressedIndexIterator<'a> {
    /// Iterator over page information
    info_iter: Box<std::slice::Iter<'a, TermBlockInformation>>,

    /// Current info
    info: Option<&'a TermBlockInformation>,

    /// Current iterator on document IDs
    docid_iterator: Option<Box<dyn Iterator<Item = DocId>>>,

    /// Current iterator on impacts
    impact_iterator: Option<Box<dyn Iterator<Item = ImpactValue>>>,

    index: usize,

    // Term index (for reference)
    term_ix: TermIndex,

    /// Our sparse index
    sparse_index: &'a CompressedIndex,
}

impl<'a> CompressedIndexIterator<'a> {
    fn new<'c: 'a>(index: &'c CompressedIndex, term_ix: TermIndex) -> Self {
        let mut iter = if term_ix < index.information.terms.len() {
            Box::new(index.information.terms[term_ix].pages.iter())
        } else {
            Box::new([].iter())
        };

        let info = iter.next();

        Self {
            // The index
            sparse_index: &index,

            // Iterator over term index page information
            info_iter: iter,

            // Current term index page information
            info: info,

            // Current docid/impact iterators
            docid_iterator: None,
            impact_iterator: None,

            // The current impact index
            index: 0,

            // Just for information purpose
            term_ix: term_ix,
        }
    }

    /// Move the iterator to the first block where a document of
    /// at least `min_doc_id` is present
    fn move_iterator(&mut self, min_doc_id: DocId) -> bool {
        // Loop until the condition is met
        while let Some(info) = self.info {
            if info.max_doc_id >= min_doc_id {
                debug!(
                    "[{}] Moving iterator OK - max(doc_id) = {} >= {}",
                    self.term_ix, info.max_doc_id, min_doc_id
                );
                return true;
            }

            // Go to the next block
            self.next_block();

            if let Some(info) = self.info {
                debug!("[{}] Read the next block (move): {}", self.term_ix, info);
            } else {
                debug!("[{}] EOF for blocks (move)", self.term_ix);
            }
        }
        false
    }

    fn read_block(&mut self, info: &TermBlockInformation) {
        let slice = self.sparse_index.docid_buffer.slice(
            info.docid_position_range.0 as usize,
            info.docid_position_range.1 as usize,
        );
        let docid_iterator = self
            .sparse_index
            .information
            .doc_ids_compressor
            .read(&mut slice.data());
        self.docid_iterator = Some(docid_iterator);

        let slice = self.sparse_index.docid_buffer.slice(
            info.value_position_range.0 as usize,
            info.value_position_range.1 as usize,
        );
        let value_iterator = self
            .sparse_index
            .information
            .values_compressor
            .read(&mut slice.data());
        self.impact_iterator = Some(value_iterator);
    }

    fn next_block(&mut self) {
        self.info = self.info_iter.next();
        self.docid_iterator = None;
        self.impact_iterator = None;
        self.index = 0;
    }
}

impl<'a> Iterator for CompressedIndexIterator<'a> {
    type Item = TermImpact;

    /// Iterate to the next doc id
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(info) = self.info {
            // We are over, load the next block
            if self.index >= info.length {
                self.next_block();
            }
            if self.info.is_none() {
                debug!("[{}] EOF for blocks", self.term_ix);
            }
        }

        if let Some(info) = self.info {
            if self.docid_iterator.is_none() {
                self.read_block(info);
                debug!("[{}] Loaded block data: {}", self.term_ix, info);
            }

            if let Some(docid) = self
                .docid_iterator
                .as_deref_mut()
                .expect("Iterator is over, but shouldn't be")
                .next()
            {
                let value = self
                    .impact_iterator
                    .as_deref_mut()
                    .expect("Impact iterator is over... but not the doc ID one")
                    .next()
                    .expect("");
                Some(TermImpact {
                    docid: docid,
                    value: value,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

struct CompressedBlockTermImpactIterator<'a> {
    iterator: RefCell<CompressedIndexIterator<'a>>,
    current_min_docid: Option<DocId>,
    // We need a RefCell for method current()
    current_value: RefCell<Option<TermImpact>>,
    max_value: ImpactValue,
    max_doc_id: DocId,
    length: usize,
}

impl<'a> CompressedBlockTermImpactIterator<'a> {
    fn new(index: &'a CompressedIndex, term_ix: TermIndex) -> Self {
        let info = &index.information.terms[term_ix];
        Self {
            iterator: RefCell::new(CompressedIndexIterator::new(index, term_ix)),
            current_value: RefCell::new(None),
            max_value: info.max_value,
            max_doc_id: info.max_doc_id,
            length: info.length,
            current_min_docid: None,
        }
    }
}

impl<'a> BlockTermImpactIterator for CompressedBlockTermImpactIterator<'a> {
    fn next_min_doc_id(&mut self, min_doc_id: DocId) -> bool {
        // Sets the current minimum document ID
        self.current_min_docid = Some(min_doc_id.max(
            if let Some(impact) = self.current_value.get_mut() {
                impact.docid + 1
            } else {
                0
            },
        ));
        let min_doc_id = self.current_min_docid.expect("Should not be None");

        // Move to the block having at least one document greater that min_doc_id
        let ok = self.iterator.get_mut().move_iterator(min_doc_id);

        if !ok {
            debug!("[{}] End of iterator", self.iterator.get_mut().term_ix)
        } else {
            debug!(
                "[{}] We have a candidate for doc_id >= {}",
                self.iterator.get_mut().term_ix,
                min_doc_id
            )
        }
        ok
    }

    /// Returns the current document ID
    fn current(&self) -> TermImpact {
        let min_docid = self.current_min_docid.expect("Should not be null");
        {
            let iterator = self.iterator.borrow();
            debug!("[{}] Searching for next {}", iterator.term_ix, min_docid);
        }

        let mut current_value = self.current_value.borrow_mut();

        if current_value
            .and_then(|x| Some(x.docid < min_docid))
            .or(Some(true))
            .unwrap()
        {
            let mut iterator = self.iterator.borrow_mut();
            debug!(
                "[{}] Current DOC ID value is {}",
                iterator.term_ix,
                if let Some(cv) = current_value.as_ref() {
                    cv.docid as i64
                } else {
                    -1
                },
            );

            *current_value = None;
            while let Some(v) = iterator.next() {
                if v.docid >= min_docid {
                    debug!("[{}] Returning {} ({})", iterator.term_ix, v.docid, v.value);

                    *current_value = Some(v);
                    break;
                }
                debug!(
                    "[{}] Skipping {} ({}) / {}",
                    iterator.term_ix, v.docid, v.value, min_docid
                );
            }

            assert!(current_value.is_some(), "Did not find current impact");
        } else {
            let iterator = self.iterator.borrow();
            debug!(
                "[{}] Current value was good {} >= {}",
                iterator.term_ix,
                current_value.expect("").docid,
                min_docid
            );
        }

        return current_value.expect("No current value");
    }

    fn max_value(&self) -> ImpactValue {
        return self.max_value;
    }

    fn max_block_doc_id(&self) -> DocId {
        self.iterator
            .borrow()
            .info
            .expect("Iterator was over")
            .max_doc_id
    }

    // fn min_block_doc_id(&self) -> DocId {
    //     self.iterator
    //         .borrow()
    //         .info
    //         .expect("Iterator was over")
    //         .min_doc_id
    // }

    fn max_block_value(&self) -> ImpactValue {
        self.iterator
            .borrow()
            .info
            .expect("Iterator was over")
            .max_value
    }

    fn max_doc_id(&self) -> DocId {
        return self.max_doc_id;
    }

    fn length(&self) -> usize {
        return self.length;
    }
}

impl BlockTermImpactIndex for CompressedIndex {
    fn iterator(
        &self,
        term_ix: crate::base::TermIndex,
    ) -> Box<dyn super::index::BlockTermImpactIterator + '_> {
        Box::new(CompressedBlockTermImpactIterator::new(self, term_ix))
    }

    fn length(&self) -> usize {
        self.information.terms.len()
    }
}

/// Compress the impact values
///
/// # Arguments
///
/// - max_block_size: maximum number of records per block
pub fn compress(
    path: &Path,
    index: &dyn BlockTermImpactIndex,
    max_block_size: usize,
    doc_ids_compressor: Box<dyn DocIdCompressor>,
    values_compressor: Box<dyn ValueCompressor>,
) -> Result<(), std::io::Error> {
    // File for impact values
    let mut value_writer = File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(path.join("value.dat"))
        .expect("Could not create the values file");

    // File for document IDs
    let mut docid_writer = File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(path.join("docid.dat"))
        .expect("Could not create the document IDs file");

    // Global information
    let mut information = CompressedIndexInformation {
        terms: Vec::new(),
        doc_ids_compressor: doc_ids_compressor,
        values_compressor: values_compressor,
    };
    let value_position = 0;
    let docid_position = 0;

    // Iterate over terms
    for term_ix in 0..index.length() {
        // Read everything
        let mut it = index.iterator(term_ix);
        let mut flag = true;

        let mut term_information = TermBlocksInformation {
            pages: Vec::new(),
            max_value: 0f32,
            max_doc_id: 0,
            length: 0,
        };

        while flag {
            // Read up to max_block_size records
            let mut impacts = Vec::new();
            let mut docids = Vec::<DocId>::new();
            flag = false;
            while let Some(ti) = it.next() {
                docids.push(ti.docid);
                impacts.push(ti.value);
                if docids.len() == max_block_size {
                    flag = true;
                    break;
                }
            }

            // Write
            information
                .doc_ids_compressor
                .write(&mut value_writer, &docids);
            information
                .values_compressor
                .write(&mut docid_writer, &impacts);

            // Add information
            let new_value_position = value_writer.stream_position()?;
            let new_docid_position = docid_writer.stream_position()?;
            let block_term_information = TermBlockInformation {
                docid_position_range: (docid_position, new_docid_position),
                value_position_range: (value_position, new_value_position),
                length: impacts.len(),
                max_value: impacts.iter().fold(0f32, |cur, x| cur.max(*x)),
                max_doc_id: docids
                    .iter()
                    .fold(0 as DocId, |cur, x| cur.max(*x))
                    .try_into()
                    .unwrap(),
            };

            term_information.max_value = term_information
                .max_value
                .max(block_term_information.max_value);
            term_information.max_doc_id = term_information
                .max_doc_id
                .max(block_term_information.max_doc_id);
            term_information.length += block_term_information.length;
            term_information.pages.push(block_term_information);
        }

        information.terms.push(term_information);
    }

    // Serialize the overall structure
    let info_path = path.join(format!("information.cbor"));
    let info_file = File::options()
        .write(true)
        .truncate(true)
        .open(info_path)
        .expect("Error while creating file");

    ciborium::ser::into_writer(&information, info_file)
        .expect("Error saving compressed term index information");

    Ok(())
}

/// Loads a block-based index
pub fn load_compressed_index(path: &Path, in_memory: bool) -> CompressedIndex {
    let info_path = path.join(format!("information.cbor"));
    let info_file = File::options()
        .read(true)
        .open(info_path)
        .expect("Error while creating file");

    let information = ciborium::de::from_reader(info_file)
        .expect("Error loading compressed term index information");

    // No load the view
    let docid_path = path.join(format!("docids.dat"));
    let impact_path = path.join(format!("impacts.dat"));
    CompressedIndex {
        information: information,
        docid_buffer: if in_memory {
            Box::new(MemoryBuffer::new(&docid_path))
        } else {
            Box::new(MmapBuffer::new(&docid_path))
        },
        impact_buffer: if in_memory {
            Box::new(MemoryBuffer::new(&impact_path))
        } else {
            Box::new(MmapBuffer::new(&impact_path))
        },
    }
}

// --- Elias Fano

struct EliasFanoCompressor {}

impl<'a> Compressor<usize> for EliasFanoCompressor {
    fn write(&self, writer: &mut dyn Write, values: &[usize]) {
        let max_value = *values.iter().max().unwrap();

        let mut c = EliasFanoBuilder::new(max_value, values.len()).expect("Error when building");

        c.append(values).expect("Could not add a doc ID");
        c.build().serialize_into(writer).expect("Yoooo");
    }

    fn read(&self, _reader: &mut dyn Read) -> Box<dyn Iterator<Item = usize>> {
        todo!()
    }
}
