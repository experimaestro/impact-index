use std::collections::HashMap;
use std::ops::DerefMut;
use std::path::Path;
use std::sync::{Arc, Mutex};

use pyo3::{pyclass, pymethods, types::PyDict, PyObject, PyRef, PyResult};
use pyo3::{IntoPy, Python};

use crate::base::{DocId, ImpactValue, TermIndex};
use crate::index::sparse::builder::{
    load_forward_index, SparseBuilderIndex, SparseBuilderIndexTrait,
};
use crate::index::sparse::maxscore::search_maxscore;
use crate::index::sparse::{
    builder::Indexer as SparseIndexer, wand::search_wand, TermImpactIterator,
    SearchFn
};

use numpy::PyArray1;

// #[pymethods]
// impl SparseBuilderIndexIterator {
//     fn __next__(mut slf: PyRefMut<Self>) -> IterNextOutput<usize, &'static str> {
//         match self.next() {
//             Some(value) => IterNextOutput::Yield(value),
//             None => IterNextOutput::Return("Ended")
//         }
//     }
// }

#[pyclass(name = "TermImpact")]
struct PyTermImpact {
    #[pyo3(get)]
    value: ImpactValue,

    #[pyo3(get)]
    docid: DocId,
}

#[pyclass]
pub struct PyScoredDocument {
    #[pyo3(get)]
    score: ImpactValue,

    #[pyo3(get)]
    docid: DocId,
}

#[pyclass]
struct SparseSparseBuilderIndexIterator {
    // Use dead code to ensure we have a valid index when iterating
    #[allow(dead_code)]
    index: Arc<Mutex<SparseBuilderIndex>>,
    iter: TermImpactIterator<'static>,
}

#[pymethods]
impl SparseSparseBuilderIndexIterator {
    fn __next__(&mut self) -> PyResult<Option<PyTermImpact>> {
        if let Some(r) = self.iter.next() {
            return Ok(Some(PyTermImpact {
                value: r.value,
                docid: r.docid,
            }));
        }
        Ok(None)
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
}

#[pyclass(name = "SparseBuilderIndex")]
pub struct PySparseBuilderIndex {
    index: Arc<Mutex<SparseBuilderIndex>>,
}

impl PySparseBuilderIndex {
    fn _search(&self, py_query: &PyDict, top_k: usize, search_fn: SearchFn) -> PyResult<PyObject> {
        let mut index = self.index.lock().unwrap();

        let query: HashMap<usize, ImpactValue> = py_query.extract()?;
        // let results = search_wand(index.deref_mut(), &query, top_k);
        let results = search_maxscore(index.deref_mut(), &query, top_k);

        let list = Python::with_gil(|py| {
            let v: Vec<PyScoredDocument> = results
                .iter()
                .map(|r| PyScoredDocument {
                    docid: r.docid,
                    score: r.score,
                })
                .collect();
            return v.into_py(py);
        });

        Ok(list)
    }
}

#[pymethods]
impl PySparseBuilderIndex {
    fn postings(&self, term: TermIndex) -> PyResult<SparseSparseBuilderIndexIterator> {
        let index = self.index.lock().unwrap();
        Ok(SparseSparseBuilderIndexIterator {
            index: self.index.clone(),
            // TODO: ugly but works since index is there
            iter: unsafe { extend_lifetime(index.iter(term)) },
        })
    }

    /// Deprecated
    fn search(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, search_wand)
    }

    fn search_wand(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, search_wand)
    }

    fn search_maxscore(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, search_maxscore)
    }

    #[staticmethod]
    fn load(folder: &str) -> PyResult<Self> {
        Ok(PySparseBuilderIndex {
            index: Arc::new(Mutex::new(load_forward_index(Path::new(folder)))),
        })
    }
}

#[pyclass(name = "SparseIndexer")]
pub struct PySparseIndexer {
    indexer: Arc<Mutex<SparseIndexer>>,
}

unsafe fn extend_lifetime<'b>(r: TermImpactIterator<'b>) -> TermImpactIterator<'static> {
    std::mem::transmute::<TermImpactIterator<'b>, TermImpactIterator<'static>>(r)
}

#[pymethods]
/// Each document is a sparse vector
impl PySparseIndexer {
    #[new]
    fn new(folder: &str) -> Self {
        PySparseIndexer {
            indexer: Arc::new(Mutex::new(SparseIndexer::new(Path::new(folder)))),
        }
    }

    /// Adds a new document to the index
    fn add(
        &mut self,
        docid: DocId,
        terms: &PyArray1<TermIndex>,
        values: &PyArray1<ImpactValue>,
    ) -> PyResult<()> {
        let mut indexer = self.indexer.lock().unwrap();
        let terms_array = unsafe { terms.as_array() };
        let values_array = unsafe { values.as_array() };
        indexer.add(docid, &terms_array, &values_array)?;
        Ok(())
    }

    fn build(&mut self) -> PyResult<PySparseBuilderIndex> {
        let mut indexer = self.indexer.lock().unwrap();
        indexer.build().expect("Error while building index");
        let index = indexer.to_forward_index();
        Ok(PySparseBuilderIndex {
            index: Arc::new(Mutex::new(index)),
        })
    }
}
