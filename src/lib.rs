
#[macro_use]
extern crate simple_error;

use std::{path::{Path}, sync::{Arc, Mutex}, borrow::BorrowMut, mem::{ManuallyDrop, transmute}};

use pyo3::{prelude::*, exceptions::PyValueError};
mod tests;
pub mod index {
    pub mod sparse;
}

use index::sparse::{Indexer as _SparseIndexer, DocId, TermIndex, ImpactValue, ForwardIndexTrait, TermImpactIterator, IndexerIterator};

use numpy::{PyArray1};


// #[pymethods] 
// impl IndexerIterator {
//     fn __next__(mut slf: PyRefMut<Self>) -> IterNextOutput<usize, &'static str> {
//         match self.next() {
//             Some(value) => IterNextOutput::Yield(value),
//             None => IterNextOutput::Return("Ended")
//         }
//     }
// }

#[pyclass(name="TermImpact")]
struct PyTermImpact {
    #[pyo3(get)]
    value: ImpactValue,

    #[pyo3(get)]
    pub docid: DocId
}

#[pyclass]
struct SparseIndexerIterator {
    indexer: Arc<Mutex<_SparseIndexer>>,
    iter: TermImpactIterator<'static>
}

#[pymethods]
impl SparseIndexerIterator {
    fn __next__(&mut self) -> PyResult<Option<PyTermImpact>> {
        if let Some(r) = self.iter.next() {
            return Ok(Some(PyTermImpact { value: r.value, docid: r.docid }))
        }
        Ok(None)
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
}

#[pyclass]
struct SparseIndexer {
    indexer: Arc<Mutex<_SparseIndexer>>,
}

unsafe fn extend_lifetime<'b>(r: TermImpactIterator<'b>) -> TermImpactIterator<'static> {
    std::mem::transmute::<TermImpactIterator<'b>, TermImpactIterator<'static>>(r)
}

#[pymethods] 
/// Each document is a sparse vector
impl SparseIndexer {
    #[new]
    fn new(folder: &str) -> Self {
        SparseIndexer { indexer: Arc::new(Mutex::new(_SparseIndexer::new(Path::new(folder)))) }
    }
    
    /// Adds a new document to the index
    fn add(&mut self, docid: DocId, terms: &PyArray1<TermIndex>, values: &PyArray1<ImpactValue>) -> PyResult<()> {
        let mut indexer  = self.indexer.lock().unwrap();        
        let terms_array = unsafe { terms.as_array() };
        let values_array = unsafe { values.as_array() };
        indexer.add(docid, &terms_array, &values_array)?;
        Ok(())
    }

    fn build(&mut self) -> PyResult<()> {
        let mut indexer  = self.indexer.lock().unwrap();        
        indexer.build().expect("yerk");
        Ok(())
    }

    fn iter(&self, term: TermIndex) -> PyResult<SparseIndexerIterator> {
        let indexer = self.indexer.lock().unwrap();
        Ok(SparseIndexerIterator {
            indexer: self.indexer.clone(),
            // TODO: ugly but works since indexer is there
            iter: unsafe { extend_lifetime(indexer.iter(term)) }
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn xpmir_rust(_py: Python, m: &PyModule) -> PyResult<()> { 
    // Index submodule
    let submod = PyModule::new(_py, "index")?;
    m.add_submodule(submod)?;

    submod.add_class::<SparseIndexer>()?;
    

    
    Ok(())
}


