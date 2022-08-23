
use pyo3::prelude::*;
mod tests;
pub mod index {
    pub mod sparse;
}

use index::sparse::{Indexer as _SparseIndexer, DocId, TermIndex, ImpactValue};

use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArray2, PyArray1, PyReadonlyArray1};

// use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pyfunction]
fn yes() -> PyResult<String> {
    Ok("Yes".to_string())
}


#[pyclass]
struct SparseIndexer {
    indexer: _SparseIndexer
}

#[pymethods] 
/// Each document is a sparse vector
impl SparseIndexer {
    #[new]
    fn new(nterms: u32) -> Self {
        SparseIndexer { indexer: _SparseIndexer::new(nterms) }
    }
    
    /// Adds a new document to the index
    fn add(&mut self, docid: DocId, terms: &PyArray1<TermIndex>, values: &PyArray1<ImpactValue>) -> PyResult<()> {
        let terms_array = unsafe { terms.as_array() };
        let values_array = unsafe { values.as_array() };
        self.indexer.add(docid, &terms_array, &values_array);
        Ok(())
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


