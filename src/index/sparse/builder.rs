use std::{path::{PathBuf, Path}, fs::File, io::{Seek, Write}};

use log::debug;
use ndarray::{ArrayBase, Ix1, Data};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use crate::{base::{TermIndex, DocId, ImpactValue, BoxResult}, index::sparse::index::TermIndexInformation};
use super::{TermImpact, TermImpactIterator, wand::{WandIterator, WandIndex}, index::{IndexInformation, TermIndexPageInformation}};


/*
* ---- First phase data structure
*
*/

struct PostingsInformation {
    postings: Vec<TermImpact>,
    information: TermIndexPageInformation
}
impl PostingsInformation {
    fn new() -> Self {
        Self {
            postings: Vec::new(),
            information: TermIndexPageInformation::new()
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.postings.len()
    }
}

struct TermsImpacts {
    in_memory_threshold: usize,
    folder: PathBuf,
    postings_file: std::fs::File,
    information: IndexInformation,
    postings_information: Vec<PostingsInformation>
}

impl TermsImpacts {
    /// Create a new term impacts in memory structure 
    fn new(folder: &Path) -> TermsImpacts {
        let path = folder.join(format!("postings.dat"));

        TermsImpacts { 
            /// Maximum number of postings (for a term, 16kb by default)
            in_memory_threshold: 16 * 1024,
            folder: folder.to_path_buf(),
            postings_file: File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true).open(path).expect("Error while creating file"),
            postings_information: Vec::new(),
            information: IndexInformation::new()
        }
    }

    /// Adds a term for a given document
    fn add_impact(&mut self, term_ix: TermIndex, docid: DocId, value: ImpactValue) -> Result<(), std::io::Error> {
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
                    max_doc_id: 0
                 });
            };
        }

        // Update the postings information
        let p_info = &mut self.postings_information[term_ix];
        
        p_info.postings.push(TermImpact {
            docid: docid,
            value: value
        });
        if p_info.information.max_value < value {
            p_info.information.max_value = value;
        }
        p_info.information.max_doc_id = docid;


        // Update the term information
        let info = &mut self.information.terms[term_ix];
        info.length += 1;

        if info.max_value < value {
            info.max_value = value
        }

        assert!(info.length == 1 || (info.max_doc_id < docid), "Doc ID should be increasing and this is not the case: {} vs {}",
            info.max_doc_id, docid);
        info.max_doc_id = docid;

        // Flush if needed
        if self.postings_information[term_ix].postings.len() > self.in_memory_threshold {
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
        debug!("Flush {} at {} (length {})", term_ix, position, len_postings);

        let mut postings_info = std::mem::replace(&mut self.postings_information[term_ix], PostingsInformation::new());
        postings_info.information.docid_position = position;
        postings_info.information.value_position = position; // will be ignored anyways
        postings_info.information.length = len_postings;
        self.information.terms[term_ix].pages.push(postings_info.information);

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
    built: bool
}

impl Indexer {
    pub fn new(folder: &Path) -> Indexer {
        Indexer { impacts: TermsImpacts::new(folder), folder: folder.to_path_buf(), built: false }
    }

    pub fn add<S, T>(&mut self, docid: DocId, terms: &ArrayBase<S, Ix1>, values: &ArrayBase<T, Ix1>) -> Result<(), std::io::Error> 
    where S: Data<Elem = TermIndex>, T: Data<Elem = ImpactValue> 
    {
        assert!(!self.built, "Index cannot be changed since it has been built");
        assert!(terms.len() == values.len(), "Value and term lists should have the same length");
        for ix in 0..terms.len() {
            self.impacts.add_impact(terms[ix], docid, values[ix])?;
        }
        Ok(())
    }

    // Closes the index structures and finishes the on-disk serialization
    pub fn build(&mut self) -> BoxResult<()> {
        if ! self.built {             
            // Flush the last impacts
            self.impacts.flush_all()?;
            self.built = true;

            let info_path = self.folder.join(format!("information.cbor"));
            let info_file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true).open(info_path).expect("Error while creating file");
            
            ciborium::ser::into_writer(&self.impacts.information, info_file).expect("Error while serializing");

        } else {
            println!("Already built")
        }

        Ok(())
    }

    /**
     * Get a forward index
     */
    pub fn to_forward_index(&mut self) -> SparseBuilderIndex {
        assert!(self.built, "Index is not built");
        assert!(self.impacts.information.terms.len() > 0, "Index has already been consumed into a forward index");
        
        let folder = self.impacts.folder.to_path_buf();
        let path = folder.as_path().join(format!("postings.dat"));
        let file = File::options()
            .read(true)
            .open(path).expect("Error while creating file");

        let mut terms = Vec::new();
        std::mem::swap(&mut self.impacts.information.terms, &mut terms);

        SparseBuilderIndex{
            terms: terms, 
            // folder: folder,
            file: file
        }
    }
}

impl<'a> SparseBuilderIndexTrait<'a> for SparseBuilderIndex {
    fn term_count(&self) -> usize {
        self.terms.len()
    }

    fn iter(&'a self, term_ix: TermIndex) -> TermImpactIterator<'a> {
        Box::new(SparseBuilderIndexIterator::new(self, term_ix))
    }
}


/// The forward index is the raw structure built while
/// constructing the index
pub struct SparseBuilderIndex {
    terms: Vec<TermIndexInformation>,
    // folder: PathBuf,
    /// postings.dat
    file: File,
}

pub fn load_forward_index(path: &Path) -> SparseBuilderIndex {
    let info_path = path.join(format!("information.cbor"));
    let info_file = File::options()
        .read(true).open(info_path).expect("Error while creating file");

    let ti : IndexInformation = ciborium::de::from_reader(info_file).expect("Error loading term index information");

    let postings_path = path.join(format!("postings.dat"));
    let file = File::options()
        .read(true)
        .open(postings_path).expect("Error while creating file");

    SparseBuilderIndex {
        terms: ti.terms,
        // folder: path.to_path_buf(),
        file
    }
}


/// Forward Index trait
pub trait SparseBuilderIndexTrait<'a> {
    fn term_count(&self) -> usize;
    fn iter(&'a self, term_ix: TermIndex) -> TermImpactIterator<'a>;
}

pub struct SparseBuilderIndexIterator<'a> {
    info_iter: Box<std::slice::Iter<'a, TermIndexPageInformation>>,
    file: &'a File,
    impacts: Vec<TermImpact>,
    index: usize,
    term_ix: TermIndex
}

impl<'a> SparseBuilderIndexIterator<'a> {
    fn new<'b: 'a>(index: &'b SparseBuilderIndex, term_ix: TermIndex) -> Self {
        let v = Vec::<TermImpact>::new();
        let iter = 
            if term_ix < index.terms.len() { 
                Box::new(index.terms[term_ix].pages.iter())
            } else { 
                Box::new([].iter()) 
            };

        Self {
            info_iter: iter,
            file: &index.file,
            impacts: v,
            index: 0,
            term_ix: term_ix
        }
    }
}

impl<'a> Iterator for SparseBuilderIndexIterator<'a> {
    type Item = TermImpact;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.impacts.len() {
            let info_option = self.info_iter.next();
            if let Some(info) = info_option {   
                self.file.seek(std::io::SeekFrom::Start(info.docid_position)).expect("Erreur de lecture");

                self.index = 0;
                self.impacts.clear();
                for _ in 0..info.length {
                    let docid: DocId = self.file.read_u64::<BigEndian>().expect(&format!("Erreur de lecture at position {}", info.docid_position));
                    let value: ImpactValue = self.file.read_f32::<BigEndian>().expect("Erreur de lecture");
                    self.impacts.push(TermImpact {
                        docid: docid,
                        value: value
                    })
                }
            } else {
                return None
            }
        }

        let index = self.index;
        self.index += 1;
        return Some(self.impacts[index]);
    }
}



// --- Wand 


struct WandSparseBuilderIndexIterator<'a> {
    iterator: SparseBuilderIndexIterator<'a>,
    current_value: TermImpact,
    max_value: ImpactValue,
    max_doc_id: DocId,
    length: usize,
}

impl<'a> WandSparseBuilderIndexIterator<'a> {
    fn new(index: &'a SparseBuilderIndex, term_ix: TermIndex) -> Self {
        let info = &index.terms[term_ix];
        Self {
            iterator: SparseBuilderIndexIterator::new(index, term_ix),
            current_value: TermImpact { docid: 0, value: 0. },
            max_value: info.max_value,
            max_doc_id: info.max_doc_id,
            length: info.length
        }
    }
}

impl<'a> WandIterator for WandSparseBuilderIndexIterator<'a> {
    fn next_min_doc_id(&mut self, doc_id: DocId) -> bool {
        while let Some(v) = self.iterator.next() {
            if v.docid >= doc_id {
                debug!("[{}] Returning {} ({})", self.iterator.term_ix, v.docid, v.value);
                self.current_value = v;
                return true
            }
            debug!("[{}] Skipping {} ({}) / {}", self.iterator.term_ix, v.docid, v.value, doc_id);
        }
        
        debug!("[{}] WAND iterator closing", self.iterator.term_ix);
        return false
    }

    fn current(&self) -> &TermImpact {
        return &self.current_value
    }

    fn max_value(&self) -> ImpactValue {
        return self.max_value
    }

    fn max_doc_id(&self) -> DocId {
        return self.max_doc_id
    }

    fn length(&self) -> usize {
        return self.length
    }

}

impl WandIndex for SparseBuilderIndex {
    fn iterator<'a>(&'a self, term_ix: TermIndex) -> Box<dyn WandIterator + 'a> {
        Box::new(WandSparseBuilderIndexIterator::new(self, term_ix))
    }

    fn length(&self) -> usize {
        return self.term_count()
    }
}
