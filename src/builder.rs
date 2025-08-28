use std::{
    cell::RefCell,
    fs::{self, File},
    io::{Seek, Write},
    path::{Path, PathBuf},
};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use derivative::Derivative;
use log::{debug, info};
use ndarray::{ArrayBase, Data, Ix1};
use serde::{Deserialize, Serialize};

use super::{
    index::{BlockTermImpactIterator, SparseIndex},
    index::{IndexInformation, TermIndexPageInformation},
};
use crate::{
    base::{BoxResult, DocId, ImpactValue, TermIndex},
    index::{SparseIndexInformation, TermIndexInformation},
};
use crate::{
    base::{Len, TermImpact},
    utils::buffer::{Buffer, MemoryBuffer, MmapBuffer, Slice},
};

/*
* ---- First phase data structure
*
*/

#[derive(Derivative, Clone)]
#[derivative(Default)]
pub struct BuilderOptions {
    /// Build a checkpoint every X documents
    /// (0 means no check-pointing)
    #[derivative(Default(value = "0"))]
    pub checkpoint_frequency: DocId,

    // Maximum number of postings For a term, 16 * 1024 postings by
    // default, that is roughly 4Gb of max. memory in total
    // with a vocabulary of 32k tokens
    #[derivative(Default(value = "16384"))]
    pub in_memory_threshold: usize,
}

/// Holds a block of impacts together
/// with the information
#[derive(Serialize, Deserialize)]
struct PostingsInformation {
    postings: Vec<TermImpact>,
    information: TermIndexPageInformation,
}
impl PostingsInformation {
    fn new() -> Self {
        Self {
            postings: Vec::new(),
            information: TermIndexPageInformation::new(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.postings.len()
    }
}

struct TermsImpacts {
    /// When using checkpointing, this is the last doc ID
    checkpoint_doc_id: Option<DocId>,
    options: BuilderOptions,
    folder: PathBuf,
    postings_file: std::fs::File,
    information: IndexInformation,
    postings_information: Vec<PostingsInformation>,
}

impl TermsImpacts {
    /**
     * Create a new term impacts in memory structure
     */
    fn new(folder: &Path, options: &BuilderOptions) -> TermsImpacts {
        let path = folder.join(format!("postings.dat"));

        let mut file_options = File::options();
        file_options.read(true).write(true).create(true);

        if options.checkpoint_frequency == 0 {
            file_options.truncate(true);
        } else {
            file_options.truncate(false);
        }

        let mut _self = TermsImpacts {
            checkpoint_doc_id: None,
            options: options.clone(),
            folder: folder.to_path_buf(),
            postings_file: file_options
                .open(path)
                .expect("Error while creating postings file."),
            postings_information: Vec::new(),
            information: IndexInformation::new(),
        };

        // Check if there is a checkpoint
        if options.checkpoint_frequency > 0 {
            let info_path = _self.folder.join(format!("checkpoint.cbor"));
            if fs::exists(&info_path).expect("error while checking if checkpoint exists") {
                let ckpt_file = File::options()
                    .read(true)
                    .open(info_path)
                    .expect("Error while opening checkpoint file");

                let pos: u64;
                (
                    _self.information,
                    _self.postings_information,
                    pos,
                    _self.checkpoint_doc_id,
                ) = ciborium::de::from_reader(ckpt_file).expect("error while reading checkpoint");

                // Note that we don't truncate the file since we will overwrite
                // everything from there
                _self
                    .postings_file
                    .seek(std::io::SeekFrom::Start(pos))
                    .expect("Error while moving in the posting file");

                info!(
                    "Read checkpoint (current doc ID {})",
                    _self.checkpoint_doc_id.unwrap()
                );
            }
        }

        _self
    }

    /// Build a checkpoint
    fn checkpoint(&mut self, doc_id: DocId) {
        info!("Check pointing index ({})", doc_id);
        self.postings_file
            .flush()
            .expect("error when flushing the posting file");

        let info_path = self.folder.join(format!("checkpoint.cbor"));
        let tmp_info_path = self.folder.join(format!("checkpoint.cbor.tmp"));
        let info_file = File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_info_path)
            .expect("Error while creating checkpoint file");

        let pos = self.postings_file.stream_position().expect("");
        ciborium::ser::into_writer(
            &(&self.information, &self.postings_information, pos, doc_id),
            info_file,
        )
        .expect("Error while serializing");

        // Move file to checkpoint
        fs::rename(tmp_info_path, info_path).expect("Error when moving checkpoint.cbor in place");

        // And then set the last checkpoint
        self.checkpoint_doc_id = Some(doc_id);
    }
    /// Adds a term for a given document
    fn add_impact(
        &mut self,
        term_ix: TermIndex,
        docid: DocId,
        value: ImpactValue,
    ) -> Result<(), std::io::Error> {
        assert!(value > 0., "Impact values should be greater than 0");

        // Adds new vectors for missing words
        if term_ix >= self.postings_information.len() {
            let d = term_ix - self.postings_information.len() + 1;
            for _i in 1..=d {
                self.postings_information.push(PostingsInformation::new());
                self.information.terms.push(TermIndexInformation {
                    pages: Vec::new(),
                    length: 0,
                    max_value: ImpactValue::NEG_INFINITY,
                    max_doc_id: 0,
                });
            }
        }

        // Update the postings information
        let p_info = &mut self.postings_information[term_ix];

        p_info.postings.push(TermImpact {
            docid: docid,
            value: value,
        });
        if p_info.information.max_value < value {
            p_info.information.max_value = value;
        }

        // Update the term information
        let info = &mut self.information.terms[term_ix];
        info.length += 1;

        if info.max_value < value {
            info.max_value = value
        }

        assert!(
            info.length == 1 || (info.max_doc_id < docid),
            "Doc ID should be increasing and this is not the case: {} vs {}",
            info.max_doc_id,
            docid
        );
        info.max_doc_id = docid;

        // Flush if needed
        if self.postings_information[term_ix].postings.len() > self.options.in_memory_threshold {
            self.flush(term_ix)?;
        }
        Ok(())
    }

    /// Flush a term into disk
    fn flush(&mut self, term_ix: TermIndex) -> Result<(), std::io::Error> {
        let len_postings = self.postings_information[term_ix].len();
        if len_postings == 0 {
            return Ok(());
        }

        // Get the stream position
        let position = self.postings_file.stream_position()?;
        debug!(
            "Flush {} at {} (length {})",
            term_ix, position, len_postings
        );

        // Starts with a fresh PostingsInformation for this term
        let mut postings_info = std::mem::replace(
            &mut self.postings_information[term_ix],
            PostingsInformation::new(),
        );
        postings_info.information.docid_position = position;
        // postings_info.information.min_doc_id = postings_info.postings[0].docid;
        postings_info.information.max_doc_id = postings_info
            .postings
            .last()
            .expect("should not be empty")
            .docid;
        postings_info.information.value_position = position; // will be ignored anyways
        postings_info.information.length = len_postings;
        self.information.terms[term_ix]
            .pages
            .push(postings_info.information);

        // outputs the postings for this term
        for ti in postings_info.postings.iter() {
            self.postings_file.write_u64::<BigEndian>(ti.docid)?;
            self.postings_file.write_f32::<BigEndian>(ti.value)?;
        }

        Ok(())
    }

    fn flush_all(&mut self) -> Result<(), std::io::Error> {
        for term_ix in 0..self.postings_information.len() {
            self.flush(term_ix)?;
        }
        self.postings_file.flush()?;
        Ok(())
    }
}

/// The indexer consumes documents and
/// build a temporary structure
pub struct Indexer {
    impacts: TermsImpacts,
    folder: PathBuf,
    built: bool,
}

impl Indexer {
    pub fn new(folder: &Path, options: &BuilderOptions) -> Indexer {
        Indexer {
            impacts: TermsImpacts::new(folder, options),
            folder: folder.to_path_buf(),
            built: false,
        }
    }

    pub fn get_checkpoint_doc_id(&self) -> Option<DocId> {
        self.impacts.checkpoint_doc_id
    }

    pub fn add<S, T>(
        &mut self,
        docid: DocId,
        terms: &ArrayBase<S, Ix1>,
        values: &ArrayBase<T, Ix1>,
    ) -> Result<(), std::io::Error>
    where
        S: Data<Elem = TermIndex>,
        T: Data<Elem = ImpactValue>,
    {
        assert!(
            !self.built,
            "Index cannot be changed since it has been built"
        );
        assert!(
            terms.len() == values.len(),
            "Value and term lists should have the same length"
        );
        for ix in 0..terms.len() {
            self.impacts.add_impact(terms[ix], docid, values[ix])?;
        }

        // Flush terms that have not been flushed for a long time (recovery)
        if (self.impacts.options.checkpoint_frequency > 0)
            && (docid
                >= self.impacts.options.checkpoint_frequency
                    + self.impacts.checkpoint_doc_id.unwrap_or(0))
        {
            // Perform a checkpoint
            self.impacts.checkpoint(docid);
        }

        Ok(())
    }

    // Closes the index structures and finishes the on-disk serialization
    pub fn build(&mut self) -> BoxResult<()> {
        if !self.built {
            // Flush the last impacts
            self.impacts.flush_all()?;
            self.built = true;

            let info_path = self.folder.join(format!("information.cbor"));
            let info_file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(info_path)
                .expect("Error while creating file");

            ciborium::ser::into_writer(&self.impacts.information, info_file)
                .expect("Error while serializing");

            // Remove old checkpoint
            for s in ["checkpoint.cbor", "checkpoint.cbor.tmp"] {
                let cpkt_path = self.folder.join(s);
                if fs::exists(&cpkt_path).expect("error while checking if checkpoint exists") {
                    fs::remove_file(&cpkt_path).expect("error while removing checkpoint file");
                }
            }
        } else {
            println!("Already built")
        }

        Ok(())
    }

    /// Returns a sparse index (the index has to be built)
    pub fn to_index(&mut self, in_memory: bool) -> SparseBuilderIndex {
        assert!(self.built, "Index is not built");
        assert!(
            self.impacts.information.terms.len() > 0,
            "Index has already been consumed into a forward index"
        );

        load_forward_index(self.impacts.folder.as_path(), in_memory)
    }
}

/// The forward index is the raw structure built while
/// constructing the index
pub struct SparseBuilderIndex {
    /// Term information
    terms: Vec<TermIndexInformation>,

    /// View on the postings
    buffer: Box<dyn Buffer>,
}

impl SparseBuilderIndex {
    fn new(terms: Vec<TermIndexInformation>, path: &PathBuf, in_memory: bool) -> Self {
        Self {
            terms: terms,
            buffer: if in_memory {
                Box::new(MemoryBuffer::new(path))
            } else {
                Box::new(MmapBuffer::new(path))
            },
        }
    }
}

pub fn load_forward_index(path: &Path, in_memory: bool) -> SparseBuilderIndex {
    let info_path = path.join(format!("information.cbor"));
    let info_file = File::options()
        .read(true)
        .open(info_path)
        .expect("Error while creating file");

    let ti: IndexInformation =
        ciborium::de::from_reader(info_file).expect("Error loading term index information");

    let postings_path = path.join(format!("postings.dat"));

    SparseBuilderIndex::new(ti.terms, &postings_path, in_memory)
}

pub struct SparseBuilderIndexIterator<'a> {
    /// Iterator over page information
    info_iter: Box<std::slice::Iter<'a, TermIndexPageInformation>>,

    /// Current info
    info: Option<&'a TermIndexPageInformation>,

    /// Current slice
    slice: Option<Box<dyn Slice + 'a>>,

    index: usize,
    /// Term index (for reference)
    term_ix: TermIndex,

    /// Our sparse index
    sparse_index: &'a SparseBuilderIndex,
}

impl<'a> SparseBuilderIndexIterator<'a> {
    fn new<'c: 'a>(index: &'c SparseBuilderIndex, term_ix: TermIndex) -> Self {
        let mut iter = if term_ix < index.terms.len() {
            Box::new(index.terms[term_ix].pages.iter())
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

            // Current memory slice (None if not loaded)
            slice: None,

            // The current impact index
            index: 0,

            // Just for information purpose
            term_ix: term_ix,
        }
    }

    /// Move the iterator to the first block where a document of
    /// at least `min_doc_id` is present
    fn move_iterator(&mut self, min_doc_id: DocId) -> Option<DocId> {
        // Loop until the condition is met
        while let Some(info) = self.info {
            if info.max_doc_id >= min_doc_id {
                debug!(
                    "[{}] Moving iterator OK - max(doc_id) = {} >= {}",
                    self.term_ix, info.max_doc_id, min_doc_id
                );
                return Some(min_doc_id);
            }

            // Go to the next block
            self.next_block();

            if let Some(info) = self.info {
                debug!("[{}] Read the next block (move): {}", self.term_ix, info);
            } else {
                debug!("[{}] EOF for blocks (move)", self.term_ix);
            }
        }

        // No result
        None
    }

    const RECORD_SIZE: usize = std::mem::size_of::<DocId>() + std::mem::size_of::<ImpactValue>();

    fn read_block(&mut self, info: &TermIndexPageInformation) {
        let start = info.docid_position as usize;
        let end = info.docid_position as usize + info.length * Self::RECORD_SIZE;

        let slice = self.sparse_index.buffer.slice(start, end);
        self.slice = Some(slice);
    }

    fn next_block(&mut self) {
        self.info = self.info_iter.next();
        self.slice = None;
        self.index = 0;
    }
}

impl<'a> Iterator for SparseBuilderIndexIterator<'a> {
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
            if self.slice.is_none() {
                self.read_block(info);
                debug!("[{}] Loading block data: {}", self.term_ix, info);
            }

            let data = self.slice.as_ref().expect("").as_ref().data();
            let start = (self.index as usize) * Self::RECORD_SIZE;
            let end = start + Self::RECORD_SIZE;
            let mut slice_ptr = &data[start..end];

            let docid: DocId = slice_ptr
                .read_u64::<BigEndian>()
                .expect("Erreur de lecture");
            let value: ImpactValue = slice_ptr
                .read_f32::<BigEndian>()
                .expect("Erreur de lecture");
            self.index += 1;
            Some(TermImpact {
                docid: docid,
                value: value,
            })
        } else {
            None
        }
    }
}

// --- Block index

struct SparseBuilderBlockTermImpactIterator<'a> {
    iterator: RefCell<SparseBuilderIndexIterator<'a>>,
    current_min_docid: Option<DocId>,
    // We need a RefCell for method current()
    current_value: RefCell<Option<TermImpact>>,
    max_value: ImpactValue,
    max_doc_id: DocId,
    length: usize,
}

impl<'a> SparseBuilderBlockTermImpactIterator<'a> {
    fn new(index: &'a SparseBuilderIndex, term_ix: TermIndex) -> Self {
        let info = &index.terms[term_ix];
        Self {
            iterator: RefCell::new(SparseBuilderIndexIterator::new(index, term_ix)),
            current_value: RefCell::new(None),
            max_value: info.max_value,
            max_doc_id: info.max_doc_id,
            length: info.length,
            current_min_docid: None,
        }
    }
}

impl<'a> BlockTermImpactIterator for SparseBuilderBlockTermImpactIterator<'a> {
    fn next_min_doc_id(&mut self, min_doc_id: DocId) -> Option<DocId> {
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
        if let Some(min_doc_id) = self.iterator.get_mut().move_iterator(min_doc_id) {
            debug!(
                "[{}] We have a candidate for doc_id >= {}",
                self.iterator.get_mut().term_ix,
                min_doc_id
            );

            Some(min_doc_id)
        } else {
            debug!("[{}] End of iterator", self.iterator.get_mut().term_ix);
            None
        }
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

impl SparseIndex for SparseBuilderIndex {
    fn block_iterator(&'_ self, term_ix: TermIndex) -> Box<dyn BlockTermImpactIterator + '_> {
        Box::new(SparseBuilderBlockTermImpactIterator::new(self, term_ix))
    }

    fn max_doc_id(&self) -> DocId {
        self.terms
            .iter()
            .map(|term| term.max_doc_id)
            .max()
            .unwrap_or(0)
    }
}

impl SparseIndexInformation for SparseBuilderIndex {
    fn value_range(&self, term_ix: TermIndex) -> (ImpactValue, ImpactValue) {
        return (0., self.terms[term_ix].max_value);
    }
}

impl Len for SparseBuilderIndex {
    fn len(&self) -> usize {
        return self.terms.len();
    }
}
