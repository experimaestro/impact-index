use log::debug;
use pyo3::{
    pyclass, pymethods, pymodule, types::PyDict, types::PyModule, IntoPy, Py, PyAny, PyObject,
    PyRef, PyResult, Python,
};
use pyo3::{PyClassInitializer, ToPyObject};

use std::collections::HashMap;
use std::future::IntoFuture;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task;

use crate::builder::BuilderOptions;
use crate::compress;
use crate::compress::docid::EliasFanoCompressor;
use crate::compress::CompressionTransform;
use crate::transforms::split::SplitIndexTransform;

use crate::base::load_index;
use crate::base::{DocId, ImpactValue, TermIndex};
use crate::index::SparseIndex;
use crate::search::maxscore::{search_maxscore, MaxScoreOptions};
use crate::transforms::IndexTransform;
use crate::{
    base::SearchFn, base::TermImpactIterator, builder::Indexer as SparseIndexer,
    search::wand::search_wand,
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

#[pyclass(name = "SparseIndexIterator")]
struct PySparseIndexIterator {
    // Use dead code to ensure we have a valid index when iterating
    #[allow(dead_code)]
    index: Arc<Box<dyn SparseIndex>>,
    iter: TermImpactIterator<'static>,
}

#[pymethods]
impl PySparseIndexIterator {
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

    fn length(&self) -> usize {
        self.iter.length()
    }

    /// Returns the term maximum impact
    fn max_value(&self) -> ImpactValue {
        self.iter.max_value()
    }

    /// Returns the maximum document ID
    fn max_doc_id(&self) -> DocId {
        self.iter.max_doc_id()
    }
}

#[pyclass(subclass, name = "IndexView")]
pub struct PyIndexView {}

#[pyclass(name = "Index", extends=PyIndexView)]
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
    fn postings(&self, term: TermIndex) -> PyResult<PySparseIndexIterator> {
        Ok(PySparseIndexIterator {
            index: self.index.clone(),
            // TODO: ugly but works since index is up here
            // Could use ouroboros::self_referencing
            iter: unsafe { extend_lifetime(self.index.block_iterator(term)) },
        })
    }

    fn num_postings(&self) -> usize {
        self.index.len()
    }

    /// Deprecated
    fn search(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, search_wand)
    }

    fn search_wand(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, search_wand)
    }

    fn search_maxscore(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, |index, query, top_k| {
            let options = MaxScoreOptions::default();
            search_maxscore(index, query, top_k, options)
        })
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
        self._aio_search(py, py_query, top_k, |index, query, top_k| {
            let options = MaxScoreOptions::default();
            search_maxscore(index, query, top_k, options)
        })
    }

    /// Convert into a BMP index
    fn to_bmp<'a>(&self, output: &str, bsize: usize, compress_range: bool) -> PyResult<()> {
        let index = self.index.clone();
        let output_path = PathBuf::from_str(output).expect("cannot use path");
        index
            .convert_to_bmp(&output_path, bsize, compress_range)
            .expect("Failed to write the CIFF file");

        Ok(())
    }

    #[staticmethod]
    fn load(py: Python<'_>, folder: &str, in_memory: bool) -> PyResult<PyObject> {
        let base = PyClassInitializer::from(PyIndexView {});
        let sub = base.add_subclass(PySparseIndex {
            index: Arc::new(load_index(Path::new(folder), in_memory)),
        });

        Ok(Py::new(py, sub)?.to_object(py))
    }
}

#[pyclass(name = "BuilderOptions")]
struct PyBuilderOptions(BuilderOptions);

#[pymethods]
impl PyBuilderOptions {
    #[new]
    fn new() -> Self {
        PyBuilderOptions {
            0: BuilderOptions::default(),
        }
    }
    #[getter]
    fn checkpoint_frequency(&self) -> DocId {
        self.0.checkpoint_frequency
    }

    #[setter]
    fn set_checkpoint_frequency(&mut self, value: DocId) {
        self.0.checkpoint_frequency = value;
    }

    #[getter]
    fn in_memory_threshold(&self) -> usize {
        self.0.in_memory_threshold
    }

    #[setter]
    fn set_in_memory_threshold(&mut self, value: usize) {
        self.0.in_memory_threshold = value;
    }
}

#[pyclass(name = "IndexBuilder")]
pub struct PyIndexBuilder {
    indexer: Arc<Mutex<SparseIndexer>>,
}

unsafe fn extend_lifetime<'b>(r: TermImpactIterator<'b>) -> TermImpactIterator<'static> {
    std::mem::transmute::<TermImpactIterator<'b>, TermImpactIterator<'static>>(r)
}

#[pymethods]
/// Each document is a sparse vector
impl PyIndexBuilder {
    #[new]
    fn new(folder: &str, options: Option<&PyBuilderOptions>) -> Self {
        let builder_options = match &options {
            Some(_options) => &_options.0,
            None => &BuilderOptions::default(),
        };

        PyIndexBuilder {
            indexer: Arc::new(Mutex::new(SparseIndexer::new(
                Path::new(folder),
                builder_options,
            ))),
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

    fn get_checkpoint_doc_id(&self) -> Option<DocId> {
        let indexer = self.indexer.blocking_lock();
        indexer.get_checkpoint_doc_id()
    }

    fn build(&mut self, py: Python<'_>, in_memory: bool) -> PyResult<PyObject> {
        let mut indexer = self.indexer.blocking_lock();
        indexer.build().expect("Error while building index");

        let base = PyClassInitializer::from(PyIndexView {});
        let index = indexer.to_index(in_memory);
        let sub = base.add_subclass(PySparseIndex {
            index: Arc::new(Box::new(index)),
        });

        Ok(Py::new(py, sub)?.to_object(py))
    }
}

#[pyclass(subclass)]
pub struct PyDocIdCompressor {
    inner: Arc<Box<dyn compress::DocIdCompressorFactory>>,
}

impl PyDocIdCompressor {}

#[pyclass(name="EliasFanoCompressor", extends=PyDocIdCompressor)]
pub struct PyEliasFanoCompressor {}

#[pymethods]
impl PyEliasFanoCompressor {
    #[new]
    fn new() -> (Self, PyDocIdCompressor) {
        (
            Self {},
            PyDocIdCompressor {
                inner: Arc::new(Box::new(EliasFanoCompressor {})),
            },
        )
    }
}

#[pyclass(name = "ImpactCompressor", subclass)]
pub struct PyImpactCompressorFactory {
    inner: Arc<Box<dyn compress::ImpactCompressorFactory>>,
}

impl PyImpactCompressorFactory {}

#[pyclass(name="ImpactQuantizer", extends=PyImpactCompressorFactory)]
pub struct PyImpactQuantizer {}

#[pymethods]
impl PyImpactQuantizer {
    #[new]
    fn new(nbits: u32, min: ImpactValue, max: ImpactValue) -> (Self, PyImpactCompressorFactory) {
        (
            PyImpactQuantizer {},
            PyImpactCompressorFactory {
                inner: Arc::new(Box::new(compress::impact::Quantizer::new(nbits, min, max))),
            },
        )
    }
}

#[pyclass(name="GlobalImpactQuantizer", extends=PyImpactCompressorFactory)]
pub struct PyGlobalQuantizerFactory {}

#[pymethods]
impl PyGlobalQuantizerFactory {
    #[new]
    fn new(nbits: u32) -> (Self, PyImpactCompressorFactory) {
        (
            PyGlobalQuantizerFactory {},
            PyImpactCompressorFactory {
                inner: Arc::new(Box::new(compress::impact::GlobalQuantizerFactory { nbits })),
            },
        )
    }
}

trait PyTransformFactory: Send + Sync {
    fn create(&self, py: Python<'_>) -> Box<dyn IndexTransform>;
}

#[pyclass(subclass)]
pub struct PyTransform {
    factory: Box<dyn PyTransformFactory>,
}

#[pymethods]
impl PyTransform {
    fn process(&self, path: &str, index: &PySparseIndex) -> PyResult<()> {
        Python::with_gil(|py| {
            let transform = self.factory.create(py);
            let view = index.index.as_view();
            transform.process(Path::new(path), view)
        })?;
        Ok(())
    }
}

struct PyCompressionTransformFactory {
    max_block_size: usize,
    doc_ids_compressor: Py<PyDocIdCompressor>,
    impacts_compressor: Py<PyImpactCompressorFactory>,
}

impl PyTransformFactory for PyCompressionTransformFactory {
    fn create(&self, py: Python<'_>) -> Box<dyn IndexTransform> {
        Box::new(CompressionTransform {
            max_block_size: self.max_block_size,
            impacts_compressor_factory: (*(*self.impacts_compressor.borrow(py)).inner).clone(),
            doc_ids_compressor_factory: (*(*self.doc_ids_compressor.borrow(py)).inner).clone(),
        })
    }
}

#[pyclass(extends=PyTransform, name="CompressionTransform")]
pub struct PyCompressionTransform {}

#[pymethods]
impl PyCompressionTransform {
    #[new]
    fn new(
        max_block_size: usize,
        doc_ids_compressor: Py<PyDocIdCompressor>,
        impacts_compressor: Py<PyImpactCompressorFactory>,
    ) -> (Self, PyTransform) {
        let factory = Box::new(PyCompressionTransformFactory {
            max_block_size: max_block_size,
            doc_ids_compressor: doc_ids_compressor,
            impacts_compressor: impacts_compressor,
        });
        (PyCompressionTransform {}, PyTransform { factory })
    }
}

struct PySplitIndexTransformFactory {
    sink: Py<PyTransform>,
    quantiles: Vec<f64>,
}
impl PyTransformFactory for PySplitIndexTransformFactory {
    fn create(&self, py: Python<'_>) -> Box<dyn IndexTransform> {
        let sink = self.sink.borrow(py).factory.create(py);

        Box::new(SplitIndexTransform {
            sink,
            quantiles: self.quantiles.clone(),
        })
    }
}

#[pyclass(name="SplitIndexTransform", extends=PyTransform)]
struct PySplitIndexTransform {}

#[pymethods]
impl PySplitIndexTransform {
    #[new]
    fn new(quantiles: Vec<f64>, sink: Py<PyTransform>) -> (Self, PyTransform) {
        let factory = Box::new(PySplitIndexTransformFactory { sink, quantiles });
        (PySplitIndexTransform {}, PyTransform { factory })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn impact_index(_py: Python, module: &PyModule) -> PyResult<()> {
    // Init logging
    pyo3_log::init();
    debug!("Loading xpmir-rust extension");

    module.add_class::<PyBuilderOptions>()?;
    module.add_class::<PyIndexBuilder>()?;
    module.add_class::<PySparseIndex>()?;
    module.add_class::<PySparseIndexIterator>()?;

    module.add_class::<PyEliasFanoCompressor>()?;
    module.add_class::<PyImpactQuantizer>()?;
    module.add_class::<PyGlobalQuantizerFactory>()?;
    module.add_class::<PyCompressionTransform>()?;
    module.add_class::<PySplitIndexTransform>()?;

    Ok(())
}
