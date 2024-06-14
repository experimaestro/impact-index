use pyo3::{
    pyclass, pymethods, types::PyDict, types::PyModule, IntoPy, Py, PyAny, PyObject, PyRef,
    PyResult, Python,
};
use std::collections::HashMap;
use std::future::IntoFuture;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task;

use crate::index::sparse::compress;

use crate::base::{DocId, ImpactValue, TermIndex};
use crate::index::sparse::index::{SparseIndex, SparseIndexAsView, SparseIndexView};
use crate::index::sparse::load_index;
use crate::index::sparse::maxscore::search_maxscore;
use crate::index::sparse::transforms::IndexTransform;
use crate::index::sparse::{
    builder::Indexer as SparseIndexer, wand::search_wand, SearchFn, TermImpactIterator,
};

use numpy::PyArray1;

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
    index: Arc<Box<dyn SparseIndex>>,
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

#[pyclass(name = "SparseIndex")]
pub struct PySparseIndex {
    index: Arc<Box<dyn SparseIndex>>,
}

impl PySparseIndex {
    fn _search(&self, py_query: &PyDict, top_k: usize, search_fn: SearchFn) -> PyResult<PyObject> {
        let query: HashMap<usize, ImpactValue> = py_query.extract()?;
        let results = search_fn(&**self.index, &query, top_k);

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

    fn _aio_search<'a>(
        &self,
        py: Python<'a>,
        py_query: &PyDict,
        top_k: usize,
        search_fn: SearchFn,
    ) -> PyResult<&'a PyAny> {
        let index = self.index.clone();

        let query: HashMap<usize, ImpactValue> = py_query.extract()?;

        let fut = async move {
            let results = task::spawn(async move {
                let r = search_fn(&**index, &query, top_k);
                r
            })
            .into_future()
            .await
            .expect("Error while searching");

            Ok(Python::with_gil(|py| {
                let v: Vec<PyScoredDocument> = results
                    .iter()
                    .map(|r| PyScoredDocument {
                        docid: r.docid,
                        score: r.score,
                    })
                    .collect();

                v.into_py(py)
            }))
        };

        pyo3_asyncio::tokio::future_into_py(py, fut)
    }
}

#[pymethods]
impl PySparseIndex {
    fn postings(&self, term: TermIndex) -> PyResult<SparseSparseBuilderIndexIterator> {
        Ok(SparseSparseBuilderIndexIterator {
            index: self.index.clone(),
            // TODO: ugly but works since index is up here
            iter: unsafe { extend_lifetime(self.index.iterator(term)) },
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

    fn aio_search_wand<'a>(
        &self,
        py: Python<'a>,
        py_query: &PyDict,
        top_k: usize,
    ) -> PyResult<&'a PyAny> {
        self._aio_search(py, py_query, top_k, search_wand)
    }

    fn aio_search_maxscore<'a>(
        &self,
        py: Python<'a>,
        py_query: &PyDict,
        top_k: usize,
    ) -> PyResult<&'a PyAny> {
        self._aio_search(py, py_query, top_k, search_maxscore)
    }

    #[staticmethod]
    fn load(folder: &str, in_memory: bool) -> PyResult<Self> {
        Ok(PySparseIndex {
            index: Arc::new(load_index(Path::new(folder), in_memory)),
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
        let mut indexer = self.indexer.blocking_lock();
        let terms_array = unsafe { terms.as_array() };
        let values_array = unsafe { values.as_array() };
        indexer.add(docid, &terms_array, &values_array)?;
        Ok(())
    }

    fn build(&mut self, in_memory: bool) -> PyResult<PySparseIndex> {
        let mut indexer = self.indexer.blocking_lock();
        indexer.build().expect("Error while building index");
        let index = indexer.to_index(in_memory);
        Ok(PySparseIndex {
            index: Arc::new(Box::new(index)),
        })
    }
}

#[pyclass(subclass)]
pub struct DocIdCompressor {
    inner: Arc<Box<dyn compress::DocIdCompressor>>,
}

impl DocIdCompressor {}

#[pyclass(extends=DocIdCompressor)]
pub struct EliasFanoCompressor {}

#[pymethods]
impl EliasFanoCompressor {
    #[new]
    fn new() -> (Self, DocIdCompressor) {
        (
            Self {},
            DocIdCompressor {
                inner: Arc::new(Box::new(compress::docid::EliasFanoCompressor {})),
            },
        )
    }
}

#[pyclass(subclass)]
pub struct ImpactCompressor {
    inner: Arc<Box<dyn compress::ImpactCompressor>>,
}

impl ImpactCompressor {}

#[pyclass(extends=ImpactCompressor)]
pub struct ImpactQuantizer {}

#[pymethods]
impl ImpactQuantizer {
    #[new]
    fn new(nbits: u32, min: ImpactValue, max: ImpactValue) -> (Self, ImpactCompressor) {
        (
            ImpactQuantizer {},
            ImpactCompressor {
                inner: Arc::new(Box::new(compress::impact::Quantizer::new(nbits, min, max))),
            },
        )
    }
}

#[pyclass(subclass)]
pub struct Transform {}

#[pyclass(extends=Transform)]
pub struct CompressionTransform {
    max_block_size: usize,
    doc_ids_compressor: Py<DocIdCompressor>,
    impacts_compressor: Py<ImpactCompressor>,
}

#[pymethods]
impl CompressionTransform {
    #[new]
    fn new(
        max_block_size: usize,
        doc_ids_compressor: Py<DocIdCompressor>,
        impacts_compressor: Py<ImpactCompressor>,
    ) -> (Self, Transform) {
        (
            CompressionTransform {
                max_block_size: max_block_size,
                doc_ids_compressor: doc_ids_compressor,
                impacts_compressor: impacts_compressor,
            },
            Transform {},
        )
    }

    fn process(&self, path: &str, index: &PySparseIndex) -> PyResult<()> {
        Python::with_gil(|py| {
            let transform = compress::CompressionTransform {
                max_block_size: self.max_block_size,
                impacts_compressor: (*(*self.impacts_compressor.borrow(py)).inner).clone(),
                doc_ids_compressor: (*(*self.doc_ids_compressor.borrow(py)).inner).clone(),
            };

            let view = SparseIndexAsView(&*(index.index));
            transform.process(Path::new(path), &view)
        })?;
        Ok(())
    }
}

pub fn init(module: &PyModule) -> PyResult<()> {
    module.add_class::<PySparseIndexer>()?;
    module.add_class::<PySparseIndex>()?;

    module.add_class::<EliasFanoCompressor>()?;
    module.add_class::<ImpactQuantizer>()?;
    module.add_class::<CompressionTransform>()?;

    Ok(())
}
