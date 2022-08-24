use std::{fmt, path::{Path, PathBuf}, fs::File};

use ndarray::{ArrayBase, Ix1, Data};
use serde::{Serialize, Deserialize};

pub type DocId = u64;
pub type TermIndex = usize;
pub type ImpactValue = f32;

/// Impact value and its document
#[derive(Serialize, Deserialize)]
struct TermImpact {
    docid: DocId,
    value: ImpactValue
}

impl std::fmt::Display for TermImpact {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({},{})", self.docid, self.value)
    }
}

#[derive(Serialize, Deserialize)]
struct InMemoryImpacts {
    postings: Vec<Vec<TermImpact>>,
}

impl InMemoryImpacts {
    fn new() -> InMemoryImpacts {
        InMemoryImpacts { 
            postings: Vec::new()
        }
    }

    fn len(&self) -> usize {
        self.postings.len()
    }
    
    fn add_impact(&mut self, termix: TermIndex, docid: DocId, value: ImpactValue) {
        // Resize if needed
        if termix >= self.postings.len() {
            let d = termix - self.postings.len() + 1;
            for _i in 1..=d {
                self.postings.push(Vec::new())
            };
        }

        // Add the posting
        self.postings[termix].push(TermImpact {
            docid: docid,
            value: value
        });


    }
}

type BoxResult<T> = Result<T,Box<dyn std::error::Error>>;

/// The set of postings
struct TermsImpacts {
    in_memory_impacts: InMemoryImpacts,
    in_memory_count: u64,
    in_memory_threshold: u64,
    folder: PathBuf,
    tempfile_count: u32,
}

impl TermsImpacts {
    /// Create a new term impacts in memory structure 
    fn new(folder: &Path) -> TermsImpacts {
        TermsImpacts { 
            in_memory_impacts: InMemoryImpacts::new(), 
            in_memory_count: 0, 
            in_memory_threshold: 100_000,
            folder: folder.to_path_buf(),
            tempfile_count: 0
        }
    }

    fn add_impact(&mut self, termix: TermIndex, docid: DocId, value: ImpactValue) {
        self.in_memory_impacts.add_impact(termix, docid, value);
        self.in_memory_count += 1;
        // Flush if needed
        if self.in_memory_count > self.in_memory_threshold {
            self.flush();
        }
    }

    fn flush(&mut self) -> BoxResult<()> {
        // Flush to disk
        let path = self.folder.join(format!("postings_{:08}.dat", self.tempfile_count));
        let file = File::create(path)?;
        ciborium::ser::into_writer(&self.in_memory_impacts, &file)?;
        file.sync_all()?;

        // Cleanup in memory structures
        self.tempfile_count += 1;
        self.in_memory_impacts = InMemoryImpacts::new();
        Ok(())
    }
}


/// The indexation
pub struct Indexer {
    impacts: TermsImpacts
}

impl Indexer {
    pub fn new(folder: &Path) -> Indexer {
        Indexer { impacts: TermsImpacts::new(folder) }
    }

    pub fn add<S, T>(&mut self, docid: DocId, terms: &ArrayBase<S, Ix1>, values: &ArrayBase<T, Ix1>) 
    where S: Data<Elem = TermIndex>, T: Data<Elem = ImpactValue>
    {
        assert!(terms.len() == values.len(), "Value and term lists should have the same length");
        for ix in 1..terms.len() {
            self.impacts.add_impact(terms[ix], docid, values[ix]) 
        }
    }
}