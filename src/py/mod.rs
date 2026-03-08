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
use crate::docstore;
use crate::transforms::split::SplitIndexTransform;

use bmp::index::posting_list::PostingListIterator;
use bmp::query::MAX_TERM_WEIGHT;
use bmp::search::b_search_verbose;

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

/// A single term impact: a (document ID, impact value) pair.
#[pyclass(name = "TermImpact")]
struct PyTermImpact {
    /// The impact value.
    #[pyo3(get)]
    value: ImpactValue,

    /// The document identifier.
    #[pyo3(get)]
    docid: DocId,
}

/// A document with its retrieval score, returned by search methods.
#[pyclass]
pub struct PyScoredDocument {
    /// The relevance score.
    #[pyo3(get)]
    score: ImpactValue,

    /// The document identifier.
    #[pyo3(get)]
    docid: DocId,
}

/// Iterator over term impacts in a posting list.
///
/// Yields TermImpact objects with (docid, value) pairs.
/// Also provides metadata: length(), max_value(), max_doc_id().
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

    /// Returns the total number of postings for this term.
    fn length(&self) -> usize {
        self.iter.length()
    }

    /// Returns the maximum impact value for this term.
    fn max_value(&self) -> ImpactValue {
        self.iter.max_value()
    }

    /// Returns the maximum document ID in this posting list.
    fn max_doc_id(&self) -> DocId {
        self.iter.max_doc_id()
    }
}

/// Base class for index views.
#[pyclass(subclass, name = "IndexView")]
pub struct PyIndexView {}

/// A loaded sparse index that supports searching and iteration.
///
/// Use ``Index.load(folder, in_memory)`` to load an existing index,
/// or build one with ``IndexBuilder``.
///
/// Example::
///
///     import impact_index
///     index = impact_index.Index.load("/path/to/index", in_memory=True)
///     results = index.search_wand({42: 1.5, 100: 0.8}, top_k=10)
///     for doc in results:
///         print(doc.docid, doc.score)
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
    /// Returns an iterator over the posting list for the given term index.
    ///
    /// Args:
    ///     term: The term index (0-based vocabulary position)
    ///
    /// Returns:
    ///     A SparseIndexIterator yielding TermImpact objects
    fn postings(&self, term: TermIndex) -> PyResult<PySparseIndexIterator> {
        Ok(PySparseIndexIterator {
            index: self.index.clone(),
            // TODO: ugly but works since index is up here
            // Could use ouroboros::self_referencing
            iter: unsafe { extend_lifetime(self.index.block_iterator(term)) },
        })
    }

    /// Returns the number of distinct terms in the index.
    fn num_postings(&self) -> usize {
        self.index.len()
    }

    /// Search the index (deprecated, use search_wand instead).
    ///
    /// Args:
    ///     py_query: Dictionary mapping term indices to query weights
    ///     top_k: Number of top results to return
    ///
    /// Returns:
    ///     List of ScoredDocument sorted by decreasing score
    fn search(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, search_wand)
    }

    /// Search using the WAND algorithm.
    ///
    /// Args:
    ///     py_query: Dictionary mapping term indices (int) to query weights (float)
    ///     top_k: Number of top results to return
    ///
    /// Returns:
    ///     List of ScoredDocument sorted by decreasing score
    fn search_wand(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, search_wand)
    }

    /// Search using the MaxScore algorithm.
    ///
    /// Args:
    ///     py_query: Dictionary mapping term indices (int) to query weights (float)
    ///     top_k: Number of top results to return
    ///
    /// Returns:
    ///     List of ScoredDocument sorted by decreasing score
    fn search_maxscore(&self, py_query: &PyDict, top_k: usize) -> PyResult<PyObject> {
        self._search(py_query, top_k, |index, query, top_k| {
            let options = MaxScoreOptions::default();
            search_maxscore(index, query, top_k, options)
        })
    }

    /// Async version of search_wand. Returns an awaitable result.
    ///
    /// Args:
    ///     py_query: Dictionary mapping term indices to query weights
    ///     top_k: Number of top results to return
    fn aio_search_wand<'a>(
        &self,
        py: Python<'a>,
        py_query: &PyDict,
        top_k: usize,
    ) -> PyResult<&'a PyAny> {
        self._aio_search(py, py_query, top_k, search_wand)
    }

    /// Async version of search_maxscore. Returns an awaitable result.
    ///
    /// Args:
    ///     py_query: Dictionary mapping term indices to query weights
    ///     top_k: Number of top results to return
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

    /// Convert the index into BMP format (legacy, loads all postings into memory).
    ///
    /// Args:
    ///     output: Output file path for the BMP index
    ///     bsize: Block size for BMP partitioning
    ///     compress_range: Whether to compress block max scores
    fn to_bmp(&self, output: &str, bsize: usize, compress_range: bool) -> PyResult<()> {
        let index = self.index.clone();
        let output_path = PathBuf::from_str(output).expect("cannot use path");
        index
            .convert_to_bmp(&output_path, bsize, compress_range)
            .expect("Failed to write the BMP file");

        Ok(())
    }

    /// Convert into a BMP index using streaming (memory-efficient) method.
    ///
    /// Uses O(num_terms * num_blocks) memory instead of O(total_postings),
    /// making it suitable for large indices that don't fit in memory.
    ///
    /// Args:
    ///     output: Output file path for the BMP index
    ///     bsize: Block size for BMP partitioning
    ///     compress_range: Whether to compress block max scores
    fn to_bmp_streaming(&self, output: &str, bsize: usize, compress_range: bool) -> PyResult<()> {
        let index = self.index.clone();
        let output_path = PathBuf::from_str(output).expect("cannot use path");
        index
            .convert_to_bmp_streaming(&output_path, bsize, compress_range)
            .expect("Failed to write the BMP file");

        Ok(())
    }

    /// Load an index from a directory.
    ///
    /// Args:
    ///     folder: Path to the index directory
    ///     in_memory: If True, loads data into RAM; otherwise uses memory-mapped I/O
    ///
    /// Returns:
    ///     An Index instance ready for searching
    #[staticmethod]
    fn load(py: Python<'_>, folder: &str, in_memory: bool) -> PyResult<PyObject> {
        let base = PyClassInitializer::from(PyIndexView {});
        let sub = base.add_subclass(PySparseIndex {
            index: Arc::new(load_index(Path::new(folder), in_memory)),
        });

        Ok(Py::new(py, sub)?.to_object(py))
    }
}

/// Configuration options for IndexBuilder.
///
/// Attributes:
///     checkpoint_frequency (int): Build a checkpoint every N documents (0 disables).
///     in_memory_threshold (int): Max postings per term before flushing to disk.
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

/// Builds a sparse index from document impact vectors.
///
/// Example::
///
///     import numpy as np
///     import impact_index
///
///     builder = impact_index.IndexBuilder("/path/to/index")
///     terms = np.array([0, 5, 42], dtype=np.uintp)
///     values = np.array([1.2, 0.5, 3.1], dtype=np.float32)
///     builder.add(0, terms, values)
///     index = builder.build(in_memory=True)
#[pyclass(name = "IndexBuilder")]
pub struct PyIndexBuilder {
    indexer: Arc<Mutex<SparseIndexer>>,
}

unsafe fn extend_lifetime<'b>(r: TermImpactIterator<'b>) -> TermImpactIterator<'static> {
    std::mem::transmute::<TermImpactIterator<'b>, TermImpactIterator<'static>>(r)
}

#[pymethods]
impl PyIndexBuilder {
    /// Create a new IndexBuilder.
    ///
    /// Args:
    ///     folder: Directory where the index files will be written
    ///     options: Optional BuilderOptions for checkpointing and memory control
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

    /// Add a document to the index.
    ///
    /// Args:
    ///     docid: Unique document identifier (must be strictly increasing)
    ///     terms: numpy array of term indices (dtype=uintp)
    ///     values: numpy array of impact values (dtype=float32, must be > 0)
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

    /// Returns the document ID from the last checkpoint, or None if no checkpoint exists.
    fn get_checkpoint_doc_id(&self) -> Option<DocId> {
        let indexer = self.indexer.blocking_lock();
        indexer.get_checkpoint_doc_id()
    }

    /// Finalize the index and return a searchable Index.
    ///
    /// Args:
    ///     in_memory: If True, the returned Index holds data in RAM
    ///
    /// Returns:
    ///     An Index instance ready for searching
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

/// Base class for document ID compressors.
#[pyclass(subclass)]
pub struct PyDocIdCompressor {
    inner: Arc<Box<dyn compress::DocIdCompressorFactory>>,
}

impl PyDocIdCompressor {}

/// Elias-Fano encoding for document ID compression.
///
/// Provides near-optimal space usage for monotonically increasing integer
/// sequences (document IDs).
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

/// Base class for impact value compressors.
#[pyclass(name = "ImpactCompressor", subclass)]
pub struct PyImpactCompressorFactory {
    inner: Arc<Box<dyn compress::ImpactCompressorFactory>>,
}

impl PyImpactCompressorFactory {}

/// Fixed-range quantizer for impact values.
///
/// Quantizes float impact values into a fixed number of bits using
/// a specified [min, max] range.
///
/// Args:
///     nbits: Number of bits for quantization (e.g., 8 for 256 levels)
///     min: Minimum impact value in the range
///     max: Maximum impact value in the range
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

/// Auto-ranging quantizer that determines min/max from the index.
///
/// Unlike ImpactQuantizer, this computes the value range automatically
/// from the global index statistics.
///
/// Args:
///     nbits: Number of bits for quantization (e.g., 8 for 256 levels)
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

/// Base class for index transforms.
///
/// Call ``process(path, index)`` to apply the transform and write
/// the result to the given directory.
#[pyclass(subclass)]
pub struct PyTransform {
    factory: Box<dyn PyTransformFactory>,
}

#[pymethods]
impl PyTransform {
    /// Apply this transform to an index, writing the result to path.
    ///
    /// Args:
    ///     path: Output directory for the transformed index
    ///     index: The source index to transform
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

/// Transform that compresses an index using block-based encoding.
///
/// Args:
///     max_block_size: Maximum number of postings per compressed block
///     doc_ids_compressor: A DocIdCompressor (e.g., EliasFanoCompressor)
///     impacts_compressor: An ImpactCompressor (e.g., ImpactQuantizer)
///
/// Example::
///
///     transform = impact_index.CompressionTransform(
///         max_block_size=128,
///         doc_ids_compressor=impact_index.EliasFanoCompressor(),
///         impacts_compressor=impact_index.GlobalImpactQuantizer(nbits=8),
///     )
///     transform.process("/path/to/compressed", index)
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

/// Transform that splits posting lists by impact quantiles.
///
/// Partitions each term's postings into sub-lists by value ranges,
/// enabling more aggressive pruning with algorithms like MaxScore.
///
/// Args:
///     quantiles: List of quantile boundaries (e.g., [0.9] splits at the 90th percentile)
///     sink: The downstream transform to apply (e.g., a CompressionTransform)
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

/// BMP (Block-Max Pruning) Searcher for fast approximate search
#[pyclass(name = "BmpSearcher")]
pub struct PyBmpSearcher {
    index: bmp::index::inverted_index::Index,
    bfwd: bmp::index::forward_index::BlockForwardIndex,
}

#[pymethods]
impl PyBmpSearcher {
    /// Load a BMP index from a file
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let path_buf = PathBuf::from_str(path).expect("Invalid path");
        let (index, bfwd) = bmp::index::from_file(path_buf).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load BMP index: {}", e))
        })?;
        Ok(PyBmpSearcher { index, bfwd })
    }

    /// Search the BMP index
    ///
    /// Args:
    ///     query: Dictionary mapping term IDs (as strings) to weights
    ///     k: Number of results to return
    ///     alpha: BMP alpha parameter (default 1.0)
    ///     beta: BMP beta parameter (default 1.0)
    ///
    /// Returns:
    ///     Tuple of (doc_ids, scores) where doc_ids are strings and scores are floats
    #[pyo3(signature = (query, k, alpha=1.0, beta=1.0))]
    fn search(
        &self,
        query: HashMap<String, f32>,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        // Find max weight for normalization
        let max_tok_weight = query
            .iter()
            .map(|p| *p.1)
            .filter(|&value| !value.is_nan())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        // Quantize query weights
        let mut quant_query: HashMap<String, u32> = HashMap::new();
        let scale: f32 = MAX_TERM_WEIGHT as f32 / max_tok_weight;
        for (key, value) in &query {
            quant_query.insert(key.clone(), (value * scale).ceil() as u32);
        }

        // Build cursors
        let cursors: Vec<PostingListIterator> = quant_query
            .iter()
            .flat_map(|(token, freq)| self.index.get_cursor(token, *freq))
            .collect();
        let wrapped_cursors = vec![cursors; 1];

        // Search
        let mut results = b_search_verbose(wrapped_cursors, &self.bfwd, k, alpha, beta, false);

        // Extract results
        let doc_lexicon = self.index.documents();
        let mut docnos: Vec<String> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for r in results[0].to_sorted_vec().iter() {
            docnos.push(doc_lexicon[r.doc_id.0 as usize].clone());
            scores.push(r.score.into());
        }
        Ok((docnos, scores))
    }

    /// Get the number of documents in the index
    fn num_documents(&self) -> usize {
        self.index.num_documents()
    }
}

// --- DocumentStore Python bindings ---

/// A stored document with key-value metadata and binary content.
#[pyclass(name = "Document")]
pub struct PyDocument {
    inner: docstore::Document,
}

#[pymethods]
impl PyDocument {
    /// The document's key-value metadata (e.g., {"docno": "DOC001"}).
    #[getter]
    fn keys(&self) -> HashMap<String, String> {
        self.inner.keys.clone()
    }

    /// The document's binary content.
    #[getter]
    fn content(&self) -> &[u8] {
        &self.inner.content
    }
}

/// Builds a compressed document store on disk.
///
/// Documents are added one at a time with key-value metadata and binary content,
/// then finalized with build().
///
/// Args:
///     folder: Directory for the document store files
///     block_size: Number of documents per compressed block (default: 4096)
///     zstd_level: Zstandard compression level (default: 3)
///
/// Example::
///
///     builder = impact_index.DocumentStoreBuilder("/path/to/store")
///     builder.add({"docno": "DOC001"}, b"document text here")
///     builder.build()
#[pyclass(name = "DocumentStoreBuilder")]
pub struct PyDocumentStoreBuilder {
    builder: Option<docstore::builder::DocumentStoreBuilder>,
}

#[pymethods]
impl PyDocumentStoreBuilder {
    #[new]
    #[pyo3(signature = (folder, block_size=4096, zstd_level=3))]
    fn new(folder: &str, block_size: usize, zstd_level: i32) -> PyResult<Self> {
        let builder =
            docstore::builder::DocumentStoreBuilder::new(Path::new(folder), block_size, zstd_level)
                .map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to create DocumentStoreBuilder: {}",
                        e
                    ))
                })?;
        Ok(Self {
            builder: Some(builder),
        })
    }

    /// Add a document to the store.
    ///
    /// Args:
    ///     keys: Dictionary of string key-value metadata
    ///     content: Binary content of the document
    fn add(&mut self, keys: HashMap<String, String>, content: &[u8]) -> PyResult<()> {
        let doc = docstore::Document {
            keys,
            content: content.to_vec(),
        };
        self.builder
            .as_mut()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
            })?
            .add(&doc)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
    }

    /// Finalize and write the document store to disk.
    ///
    /// Can only be called once. Raises RuntimeError if called again.
    fn build(&mut self) -> PyResult<()> {
        let builder = self.builder.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Builder already consumed by build()")
        })?;
        builder
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
    }
}

/// A compressed document store for retrieving documents by number or key.
///
/// Load with ``DocumentStore.load(folder)`` and retrieve documents
/// using ``get_by_number()`` or ``get_by_key()``. Async variants
/// (``aio_get_by_number``, ``aio_get_by_key``) are also available.
///
/// Example::
///
///     store = impact_index.DocumentStore.load("/path/to/store")
///     docs = store.get_by_number([0, 1, 2])
///     for doc in docs:
///         print(doc.keys, doc.content)
#[pyclass(name = "DocumentStore")]
pub struct PyDocumentStore {
    store: Arc<docstore::store::DocumentStore>,
}

#[pymethods]
impl PyDocumentStore {
    /// Load a document store from disk.
    ///
    /// Args:
    ///     folder: Path to the document store directory
    ///     content_access: How to access content data - "memory" (load into RAM),
    ///         "mmap" (memory-mapped), or "disk" (read from disk on demand)
    ///
    /// Returns:
    ///     A DocumentStore instance
    #[staticmethod]
    #[pyo3(signature = (folder, content_access="memory"))]
    fn load(folder: &str, content_access: &str) -> PyResult<Self> {
        let access = match content_access {
            "memory" => docstore::store::ContentAccess::Memory,
            "mmap" => docstore::store::ContentAccess::Mmap,
            "disk" => docstore::store::ContentAccess::Disk,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown content_access '{}', expected 'memory', 'mmap', or 'disk'",
                    other
                )));
            }
        };
        let store = docstore::store::DocumentStore::load(Path::new(folder), access)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self {
            store: Arc::new(store),
        })
    }

    /// Returns the total number of documents in the store.
    fn num_documents(&self) -> u64 {
        self.store.num_documents()
    }

    /// Returns the list of key names defined in the store.
    fn key_names(&self) -> Vec<String> {
        self.store.key_names().to_vec()
    }

    /// Retrieve documents by their sequential number (0-based).
    ///
    /// Args:
    ///     doc_numbers: List of document numbers to retrieve
    ///
    /// Returns:
    ///     List of Document objects
    fn get_by_number(&self, doc_numbers: Vec<u64>) -> PyResult<Vec<PyDocument>> {
        let docs = self
            .store
            .get_by_number(&doc_numbers)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(docs.into_iter().map(|d| PyDocument { inner: d }).collect())
    }

    /// Retrieve documents by a key field value.
    ///
    /// Args:
    ///     key_name: Name of the key field to search (e.g., "docno")
    ///     key_values: List of key values to look up
    ///
    /// Returns:
    ///     List of Optional[Document] (None for keys not found)
    fn get_by_key(
        &self,
        key_name: &str,
        key_values: Vec<String>,
    ) -> PyResult<Vec<Option<PyDocument>>> {
        let refs: Vec<&str> = key_values.iter().map(|s| s.as_str()).collect();
        let docs = self
            .store
            .get_by_key(key_name, &refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(docs
            .into_iter()
            .map(|opt| opt.map(|d| PyDocument { inner: d }))
            .collect())
    }

    /// Async version of get_by_number. Returns an awaitable result.
    fn aio_get_by_number<'a>(&self, py: Python<'a>, doc_numbers: Vec<u64>) -> PyResult<&'a PyAny> {
        let store = self.store.clone();
        let fut = async move {
            let docs = task::spawn_blocking(move || {
                store.get_by_number(&doc_numbers).map_err(|e| e.to_string())
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Ok(Python::with_gil(|py| {
                let v: Vec<PyDocument> =
                    docs.into_iter().map(|d| PyDocument { inner: d }).collect();
                v.into_py(py)
            }))
        };
        pyo3_asyncio::tokio::future_into_py(py, fut)
    }

    /// Async version of get_by_key. Returns an awaitable result.
    fn aio_get_by_key<'a>(
        &self,
        py: Python<'a>,
        key_name: String,
        key_values: Vec<String>,
    ) -> PyResult<&'a PyAny> {
        let store = self.store.clone();
        let fut = async move {
            let docs = task::spawn_blocking(move || {
                let refs: Vec<&str> = key_values.iter().map(|s| s.as_str()).collect();
                store
                    .get_by_key(&key_name, &refs)
                    .map_err(|e| e.to_string())
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Ok(Python::with_gil(|py| {
                let v: Vec<Option<PyDocument>> = docs
                    .into_iter()
                    .map(|opt| opt.map(|d| PyDocument { inner: d }))
                    .collect();
                v.into_py(py)
            }))
        };
        pyo3_asyncio::tokio::future_into_py(py, fut)
    }
}

/// Python module for sparse index construction, compression, and search.
///
/// Provides classes for building, transforming, and querying sparse indices
/// from neural information retrieval models, as well as a compressed
/// document store.
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
    module.add_class::<PyBmpSearcher>()?;

    module.add_class::<PyDocument>()?;
    module.add_class::<PyDocumentStoreBuilder>()?;
    module.add_class::<PyDocumentStore>()?;

    Ok(())
}
