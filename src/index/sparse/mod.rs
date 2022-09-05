use std::{fmt, path::{Path, PathBuf}, fs::File, io::{Seek, Write, Read}, mem::size_of, os::unix::prelude::FileExt, vec, borrow::BorrowMut};
use byteorder::{BigEndian, WriteBytesExt, ReadBytesExt};
use ndarray::{ArrayBase, Ix1, Data};
use serde::{Serialize, Deserialize};

use crate::search::DocId;

use self::daat::WandIterator;

pub type TermIndex = usize;
pub type ImpactValue = f32;
pub mod daat;

/// Impact value and its document
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct TermImpact {
    pub docid: DocId,
    pub value: ImpactValue
}

impl std::fmt::Display for TermImpact {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({},{})", self.docid, self.value)
    }
}


#[derive(Serialize, Deserialize)]
struct TermIndexPageInformation {
    position: u64,
    length: usize
}

#[derive(Serialize, Deserialize)]
struct TermIndexInformation {
    pages: Vec<TermIndexPageInformation>,
    max_value: f64
}

/// Globla information on the index structure
#[derive(Serialize, Deserialize)]
struct IndexInformation {
    terms: Vec<TermIndexInformation>
}


impl IndexInformation {
    pub fn new() -> IndexInformation{
        IndexInformation {
            terms: Vec::new()
        }
    }
}

type BoxResult<T> = Result<T,Box<dyn std::error::Error>>;



/*
* ---- First phase data structure
*
*/
struct TermsImpacts {
    in_memory_threshold: usize,
    folder: PathBuf,
    postings_file: std::fs::File,
    information: IndexInformation,
    postings: Vec<Vec<TermImpact>>
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
            postings: Vec::new(),
            information: IndexInformation::new()
        }
    }

    fn add_impact(&mut self, termix: TermIndex, docid: DocId, value: ImpactValue) -> Result<(), std::io::Error> {
        if termix >= self.postings.len() {
            let d = termix - self.postings.len() + 1;
            for _i in 1..=d {
                self.postings.push(Vec::new());
                self.information.terms.push(TermIndexInformation { 
                    pages: Vec::new(),
                    max_value: f64::NEG_INFINITY
                 });
            };
        }

        // Add the posting
        self.postings[termix].push(TermImpact {
            docid: docid,
            value: value
        });

        if self.information.terms[termix].max_value < value as f64 {
            self.information.terms[termix].max_value = value as f64
        }


        // Flush if needed
        if self.postings[termix].len() > self.in_memory_threshold {
            self.flush(termix)?;
        }
        Ok(())
    }

    
    fn flush(&mut self, termix: TermIndex) -> Result<(), std::io::Error> {
        let len_postings = self.postings[termix].len();
        if len_postings == 0 { 
            return Ok(());
        }
      
        // Get the stream position
        self.information.terms[termix].pages.push(TermIndexPageInformation {
            position: self.postings_file.stream_position()?,
            length: len_postings,
        });

        // outputs the postings for this term
        for ti in self.postings[termix].iter() {
            self.postings_file.write_i64::<BigEndian>(ti.docid)?;
            self.postings_file.write_f32::<BigEndian>(ti.value)?;
        }

        // Cleanup in memory structures
        self.postings[termix].clear();
        Ok(())
    }

    fn flush_all(&mut self) -> Result<(), std::io::Error> {
        for term_ix in 0..self.postings.len() {
            self.flush(term_ix)?;
        }
        self.postings_file.flush()?;
        Ok(())
    }


}


pub struct ForwardIndex {
    terms: Vec<TermIndexInformation>,
    folder: PathBuf
}

/// The indexation
pub struct Indexer {
    impacts: TermsImpacts,
    built: bool
}

impl Indexer {
    pub fn new(folder: &Path) -> Indexer {
        Indexer { impacts: TermsImpacts::new(folder), built: false }
    }

    pub fn add<S, T>(&mut self, docid: DocId, terms: &ArrayBase<S, Ix1>, values: &ArrayBase<T, Ix1>) -> Result<(), std::io::Error> 
    where S: Data<Elem = TermIndex>, T: Data<Elem = ImpactValue> 
    {
        assert!(!self.built, "Index cannot be changed since it has been built");
        assert!(terms.len() == values.len(), "Value and term lists should have the same length");
        for ix in 1..terms.len() {
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
        } else {
            println!("Already built")
        }

        Ok(())
    }

    /**
     * Get a forward index
     */
    pub fn to_forward_index(self) -> ForwardIndex {
        assert!(self.built, "Index is not built");

        ForwardIndex{
            terms: self.impacts.information.terms, 
            folder: self.impacts.folder.to_path_buf()
        }
    }
}


pub type TermImpactIterator<'a> = Box<dyn Iterator<Item=TermImpact> + 'a + Send>;

/// Forward Index trait
pub trait ForwardIndexTrait<'a> {
    fn term_count(&self) -> usize;
    fn iter(&'a self, term_ix: TermIndex) -> TermImpactIterator<'a>;
}

pub struct IndexerIterator<'a> {
    info_iter: std::slice::Iter<'a, TermIndexPageInformation>,
    file: &'a File,
    impacts: Vec<TermImpact>,
    index: usize,
    term_ix: TermIndex
}

impl<'a> IndexerIterator<'a> {
    fn new(indexer: &'a Indexer, term_ix: TermIndex) -> Self {
        let v = Vec::<TermImpact>::new();
        Self {
            info_iter: indexer.impacts.information.terms[term_ix].pages.iter(),
            file: &indexer.impacts.postings_file,
            impacts: v,
            index: 0,
            term_ix: term_ix
        }
    }
}

impl<'a> Iterator for IndexerIterator<'a> {
    type Item = TermImpact;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.impacts.len() {
            let info_option = self.info_iter.next();
            if let Some(info) = info_option {
                let size = info.length * size_of::<TermImpact>();
    
                self.file.seek(std::io::SeekFrom::Start(info.position)).expect("Erreur de lecture");

                self.index = 0;
                self.impacts.clear();
                for _ in 0..size {
                    let docid: DocId = self.file.read_i64::<BigEndian>().expect("Erreur de lecture");
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

impl<'a> ForwardIndexTrait<'a> for Indexer {
    fn term_count(&self) -> usize {
        return self.impacts.information.terms.len()
    }

    fn iter(&'a self, term_ix: TermIndex) -> TermImpactIterator<'a> {
        Box::new(IndexerIterator::new(&self, term_ix))
    }
}


// Wand iteration

struct WandIndexerIterator<'a> {
    iterator: IndexerIterator<'a>,
    current_value: TermImpact,
    max_value: f64
}

impl<'a> WandIndexerIterator<'a> {
    fn new(indexer: &'a Indexer, term_ix: TermIndex) -> Self {
        Self {
            iterator: IndexerIterator::new(indexer, term_ix),
            current_value: TermImpact { docid: 0, value: 0. },
            max_value: indexer.impacts.information.terms[term_ix].max_value
        }
    }
}

impl<'a> WandIterator<'a> for WandIndexerIterator<'a> {
    fn next(&mut self, doc_id: DocId) -> bool {
        while let Some(v) = self.iterator.next() {
            println!("[{}] Got doc {} with value {} (min {})", self.iterator.term_ix, v.docid, v.value, doc_id);
            if v.docid >= doc_id {
                self.current_value = v;
                return true
            }
        }
        
        println!("[{}] This is over", self.iterator.term_ix);
        return false
    }

    fn current(&self) -> &TermImpact {
        return &self.current_value
    }

    fn max(&self) -> f64 {
        return self.max_value
    }
}

impl<'a> daat::WandIndex<'a> for Indexer {
    fn iterator(&'a self, term_ix: TermIndex) -> Box<dyn WandIterator + 'a> {
        Box::new(WandIndexerIterator::new(&self, term_ix))
    }
}