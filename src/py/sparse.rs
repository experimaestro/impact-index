use std::collections::HashMap;
use std::ops::DerefMut;
use std::path::Path;
use std::sync::{Mutex, Arc};

use pyo3::{Python, IntoPy};
use pyo3::{pyclass, pymethods, PyResult, PyRef, types::PyDict, PyObject};

use crate::base::{ImpactValue, DocId, TermIndex};
use crate::index::sparse::builder::{ForwardIndexTrait, ForwardIndex};
use crate::index::sparse::{builder::Indexer as _SparseIndexer, TermImpactIterator, wand::search_wand};

use numpy::{PyArray1};

// #[pymethods] 
// impl ForwardIndexIterator {
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
    docid: DocId
}

#[pyclass]
pub struct PyScoredDocument {
    #[pyo3(get)]
    score: f64,

    #[pyo3(get)]
    docid: DocId
}

#[pyclass]
struct SparseForwardIndexIterator {
    index: Arc<Mutex<ForwardIndex>>,
    iter: TermImpactIterator<'static>
}

#[pymethods]
impl SparseForwardIndexIterator {
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

#[pyclass(name="ForwardIndex")]
pub struct PyForwardIndex {
    index: Arc<Mutex<ForwardIndex>>,
}

#[pymethods]
impl PyForwardIndex {
    fn iter(&self, term: TermIndex) -> PyResult<SparseForwardIndexIterator> {
        let index = self.index.lock().unwrap();
        Ok(SparseForwardIndexIterator {
            index: self.index.clone(),
            // TODO: ugly but works since index is there
            iter: unsafe { extend_lifetime(index.iter(term)) }
        })
    }

    fn search(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        let mut index  = self.index.lock().unwrap();        

        let query: HashMap<usize, f64> = py_query.extract()?;
        let results = search_wand(index.deref_mut(), &query, top_k);

        let list = Python::with_gil(|py| {
            let v: Vec<PyScoredDocument> = results.iter().map(|r| PyScoredDocument {
                docid: r.docid,
                score: r.score
            }).collect();
            return v.into_py(py);
        });

        Ok(list)
    }
}


#[pyclass]
pub struct SparseIndexer {
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

    fn build(&mut self) -> PyResult<PyForwardIndex> {
        let mut indexer  = self.indexer.lock().unwrap();        
        indexer.build().expect("Error while building index");
        let index = indexer.to_forward_index();
        Ok(PyForwardIndex { index: Arc::new(Mutex::new(index)) })
    }
}