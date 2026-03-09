"""Type stubs for the impact_index module.

A Python library for efficient sparse retrieval from neural information
retrieval systems, built on Rust with PyO3 bindings.
"""

from __future__ import annotations

from typing import Awaitable, Optional, Union

import numpy as np
import numpy.typing as npt

DType = Union[str, np.dtype, type]
"""Accepted dtype specifications: string names (``"float32"``, ``"float16"``,
``"bfloat16"``, ``"float64"``, ``"int32"``, ``"int64"``), numpy dtype objects
(``np.dtype("float16")``), or numpy type classes (``np.float32``)."""


class TermImpact:
    """A single term impact: a (document ID, impact value) pair."""

    @property
    def docid(self) -> int:
        """The document identifier."""
        ...

    @property
    def value(self) -> float:
        """The impact value."""
        ...


class ScoredDocument:
    """A document with its retrieval score, returned by search methods."""

    @property
    def docid(self) -> int:
        """The document identifier."""
        ...

    @property
    def score(self) -> float:
        """The relevance score."""
        ...


class SparseIndexIterator:
    """Iterator over term impacts in a posting list.

    Yields :class:`TermImpact` objects. Also provides metadata via
    :meth:`length`, :meth:`max_value`, and :meth:`max_doc_id`.
    """

    def __iter__(self) -> SparseIndexIterator: ...
    def __next__(self) -> TermImpact: ...

    def length(self) -> int:
        """Returns the total number of postings for this term."""
        ...

    def max_value(self) -> float:
        """Returns the maximum impact value for this term."""
        ...

    def max_doc_id(self) -> int:
        """Returns the maximum document ID in this posting list."""
        ...


class IndexView:
    """Base class for index views."""

    ...


class Index(IndexView):
    """A loaded sparse index that supports searching and iteration.

    Use :meth:`load` to load an existing index from disk, or build one
    with :class:`IndexBuilder`.

    Example::

        import impact_index
        index = impact_index.Index.load("/path/to/index", in_memory=True)
        results = index.search_wand({42: 1.5, 100: 0.8}, top_k=10)
        for doc in results:
            print(doc.docid, doc.score)
    """

    @staticmethod
    def load(folder: str, in_memory: bool) -> Index:
        """Load an index from a directory.

        Args:
            folder: Path to the index directory.
            in_memory: If ``True``, loads data into RAM; otherwise uses
                memory-mapped I/O.

        Returns:
            An :class:`Index` instance ready for searching.
        """
        ...

    def postings(self, term: int) -> SparseIndexIterator:
        """Returns an iterator over the posting list for the given term.

        Args:
            term: The term index (0-based vocabulary position).

        Returns:
            A :class:`SparseIndexIterator` yielding :class:`TermImpact` objects.
        """
        ...

    def num_postings(self) -> int:
        """Returns the number of distinct terms in the index."""
        ...

    def search(self, py_query: dict[int, float], top_k: int) -> list[ScoredDocument]:
        """Search the index (deprecated, use :meth:`search_wand` instead).

        Args:
            py_query: Dictionary mapping term indices to query weights.
            top_k: Number of top results to return.

        Returns:
            List of :class:`ScoredDocument` sorted by decreasing score.
        """
        ...

    def search_wand(self, py_query: dict[int, float], top_k: int) -> list[ScoredDocument]:
        """Search using the WAND algorithm.

        Args:
            py_query: Dictionary mapping term indices to query weights.
            top_k: Number of top results to return.

        Returns:
            List of :class:`ScoredDocument` sorted by decreasing score.
        """
        ...

    def search_maxscore(self, py_query: dict[int, float], top_k: int) -> list[ScoredDocument]:
        """Search using the MaxScore algorithm.

        Args:
            py_query: Dictionary mapping term indices to query weights.
            top_k: Number of top results to return.

        Returns:
            List of :class:`ScoredDocument` sorted by decreasing score.
        """
        ...

    def aio_search_wand(
        self, py_query: dict[int, float], top_k: int
    ) -> Awaitable[list[ScoredDocument]]:
        """Async version of :meth:`search_wand`.

        Args:
            py_query: Dictionary mapping term indices to query weights.
            top_k: Number of top results to return.
        """
        ...

    def aio_search_maxscore(
        self, py_query: dict[int, float], top_k: int
    ) -> Awaitable[list[ScoredDocument]]:
        """Async version of :meth:`search_maxscore`.

        Args:
            py_query: Dictionary mapping term indices to query weights.
            top_k: Number of top results to return.
        """
        ...

    def to_bmp(self, output: str, bsize: int, compress_range: bool) -> None:
        """Convert the index into BMP format (loads all postings into memory).

        Args:
            output: Output file path for the BMP index.
            bsize: Block size for BMP partitioning.
            compress_range: Whether to compress block max scores.
        """
        ...

    def to_bmp_streaming(self, output: str, bsize: int, compress_range: bool) -> None:
        """Convert into BMP format using streaming (memory-efficient) method.

        Uses ``O(num_terms * num_blocks)`` memory instead of ``O(total_postings)``.

        Args:
            output: Output file path for the BMP index.
            bsize: Block size for BMP partitioning.
            compress_range: Whether to compress block max scores.
        """
        ...


class BuilderOptions:
    """Configuration options for :class:`IndexBuilder`.

    Attributes:
        checkpoint_frequency: Build a checkpoint every N documents
            (0 disables checkpointing).
        in_memory_threshold: Max postings per term before flushing to disk.
    """

    def __init__(self) -> None: ...

    @property
    def checkpoint_frequency(self) -> int: ...
    @checkpoint_frequency.setter
    def checkpoint_frequency(self, value: int) -> None: ...

    @property
    def in_memory_threshold(self) -> int: ...
    @in_memory_threshold.setter
    def in_memory_threshold(self, value: int) -> None: ...


class IndexBuilder:
    """Builds a sparse index from document impact vectors.

    Example::

        import numpy as np
        import impact_index

        builder = impact_index.IndexBuilder("/path/to/index")
        terms = np.array([0, 5, 42], dtype=np.uintp)
        values = np.array([1.2, 0.5, 3.1], dtype=np.float32)
        builder.add(0, terms, values)
        index = builder.build(in_memory=True)
    """

    def __init__(
        self,
        folder: str,
        options: Optional[BuilderOptions] = None,
        dtype: Optional[DType] = None,
    ) -> None:
        """Create a new IndexBuilder.

        Args:
            folder: Directory where the index files will be written.
            options: Optional :class:`BuilderOptions` for checkpointing
                and memory control.
            dtype: Value type for on-disk storage. Accepts string names
                (``"float32"``, ``"float16"``, ``"bfloat16"``, ``"float64"``,
                ``"int32"``, ``"int64"``), numpy dtype objects, or numpy type
                classes (e.g., ``np.float16``). Default is ``"float32"``.
        """
        ...

    def add(
        self,
        docid: int,
        terms: npt.NDArray[np.uintp],
        values: npt.NDArray,
    ) -> None:
        """Add a document to the index.

        Args:
            docid: Unique document identifier (must be strictly increasing).
            terms: numpy array of term indices (``dtype=np.uintp``).
            values: numpy array of impact values (any numeric dtype,
                must be > 0). Values are converted to the builder's
                dtype automatically.
        """
        ...

    def get_checkpoint_doc_id(self) -> Optional[int]:
        """Returns the document ID from the last checkpoint, or ``None``."""
        ...

    def build(self, in_memory: bool) -> Index:
        """Finalize the index and return a searchable :class:`Index`.

        Args:
            in_memory: If ``True``, the returned Index holds data in RAM.

        Returns:
            An :class:`Index` instance ready for searching.
        """
        ...


# --- Compression ---


class EliasFanoCompressor:
    """Elias-Fano encoding for document ID compression.

    Provides near-optimal space usage for monotonically increasing
    integer sequences (document IDs).
    """

    def __init__(self) -> None: ...


class ImpactQuantizer:
    """Fixed-range quantizer for impact values.

    Quantizes float impact values into a fixed number of bits using
    a specified ``[min, max]`` range.

    Args:
        nbits: Number of bits for quantization (e.g., 8 for 256 levels).
        min: Minimum impact value in the range.
        max: Maximum impact value in the range.
    """

    def __init__(self, nbits: int, min: float, max: float) -> None: ...


class GlobalImpactQuantizer:
    """Auto-ranging quantizer that determines min/max from the index.

    Unlike :class:`ImpactQuantizer`, this computes the value range
    automatically from the global index statistics.

    Args:
        nbits: Number of bits for quantization (e.g., 8 for 256 levels).
    """

    def __init__(self, nbits: int) -> None: ...


# --- Transforms ---


class CompressionTransform:
    """Transform that compresses an index using block-based encoding.

    Example::

        transform = impact_index.CompressionTransform(
            max_block_size=128,
            doc_ids_compressor=impact_index.EliasFanoCompressor(),
            impacts_compressor=impact_index.GlobalImpactQuantizer(nbits=8),
        )
        transform.process("/path/to/compressed", index)

    Args:
        max_block_size: Maximum number of postings per compressed block.
        doc_ids_compressor: A document ID compressor (e.g.,
            :class:`EliasFanoCompressor`).
        impacts_compressor: An impact value compressor (e.g.,
            :class:`GlobalImpactQuantizer`).
    """

    def __init__(
        self,
        max_block_size: int,
        doc_ids_compressor: EliasFanoCompressor,
        impacts_compressor: ImpactQuantizer | GlobalImpactQuantizer,
    ) -> None: ...

    def process(self, path: str, index: Index) -> None:
        """Apply this transform to an index, writing the result to *path*.

        Args:
            path: Output directory for the transformed index.
            index: The source index to transform.
        """
        ...


class SplitIndexTransform:
    """Transform that splits posting lists by impact quantiles.

    Partitions each term's postings into sub-lists by value ranges,
    enabling more aggressive pruning with the MaxScore algorithm.

    Args:
        quantiles: Quantile boundaries (e.g., ``[0.9]`` splits at the
            90th percentile).
        sink: The downstream transform to apply (e.g., a
            :class:`CompressionTransform`).
    """

    def __init__(
        self, quantiles: list[float], sink: CompressionTransform
    ) -> None: ...

    def process(self, path: str, index: Index) -> None:
        """Apply this transform to an index, writing the result to *path*.

        Args:
            path: Output directory for the transformed index.
            index: The source index to transform.
        """
        ...


# --- BMP ---


class BmpSearcher:
    """BMP (Block-Max Pruning) searcher for fast approximate search.

    Args:
        path: Path to a BMP index file.
    """

    def __init__(self, path: str) -> None:
        """Load a BMP index from a file.

        Args:
            path: Path to the BMP index file.
        """
        ...

    def search(
        self,
        query: dict[str, float],
        k: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> tuple[list[str], list[float]]:
        """Search the BMP index.

        Args:
            query: Dictionary mapping term IDs (strings) to weights.
            k: Number of results to return.
            alpha: BMP alpha parameter.
            beta: BMP beta parameter.

        Returns:
            Tuple of ``(doc_ids, scores)`` where doc_ids are strings
            and scores are floats.
        """
        ...

    def num_documents(self) -> int:
        """Returns the number of documents in the index."""
        ...


# --- Document Store ---


class Document:
    """A stored document with key-value metadata and binary content."""

    @property
    def internal_id(self) -> int:
        """The internal sequential ID (0-based) assigned by the document store."""
        ...

    @property
    def keys(self) -> dict[str, str]:
        """The document's key-value metadata (e.g., ``{"docno": "DOC001"}``)."""
        ...

    @property
    def content(self) -> bytes:
        """The document's binary content."""
        ...


class DocumentStoreBuilder:
    """Builds a compressed document store on disk.

    Documents are added one at a time with key-value metadata and binary
    content, then finalized with :meth:`build`.

    Example::

        builder = impact_index.DocumentStoreBuilder("/path/to/store")
        builder.add({"docno": "DOC001"}, b"document text here")
        builder.build()

    Args:
        folder: Directory for the document store files.
        block_size: Number of documents per compressed block (default: 4096).
        zstd_level: Zstandard compression level (default: 3).
    """

    def __init__(
        self, folder: str, block_size: int = 4096, zstd_level: int = 3
    ) -> None: ...

    def add(self, keys: dict[str, str], content: bytes) -> None:
        """Add a document to the store.

        Args:
            keys: Dictionary of string key-value metadata.
            content: Binary content of the document.
        """
        ...

    def build(self) -> None:
        """Finalize and write the document store to disk.

        Can only be called once. Raises ``RuntimeError`` if called again.
        """
        ...


class DocumentStore:
    """A compressed document store for retrieving documents by number or key.

    Load with :meth:`load` and retrieve documents using
    :meth:`get_by_number` or :meth:`get_by_key`. Async variants
    (:meth:`aio_get_by_number`, :meth:`aio_get_by_key`) are also available.

    Example::

        store = impact_index.DocumentStore.load("/path/to/store")
        docs = store.get_by_number([0, 1, 2])
        for doc in docs:
            print(doc.keys, doc.content)
    """

    @staticmethod
    def load(folder: str, content_access: str = "memory") -> DocumentStore:
        """Load a document store from disk.

        Args:
            folder: Path to the document store directory.
            content_access: How to access content data —
                ``"memory"`` (load into RAM),
                ``"mmap"`` (memory-mapped), or
                ``"disk"`` (read on demand).

        Returns:
            A :class:`DocumentStore` instance.
        """
        ...

    def num_documents(self) -> int:
        """Returns the total number of documents in the store."""
        ...

    def key_names(self) -> list[str]:
        """Returns the list of key names defined in the store."""
        ...

    def get_by_number(self, doc_numbers: list[int]) -> list[Document]:
        """Retrieve documents by their sequential number (0-based).

        Args:
            doc_numbers: List of document numbers to retrieve.

        Returns:
            List of :class:`Document` objects.
        """
        ...

    def get_by_key(
        self, key_name: str, key_values: list[str]
    ) -> list[Optional[Document]]:
        """Retrieve documents by a key field value.

        Args:
            key_name: Name of the key field (e.g., ``"docno"``).
            key_values: List of key values to look up.

        Returns:
            List of :class:`Document` or ``None`` for keys not found.
        """
        ...

    def aio_get_by_number(
        self, doc_numbers: list[int]
    ) -> Awaitable[list[Document]]:
        """Async version of :meth:`get_by_number`."""
        ...

    def aio_get_by_key(
        self, key_name: str, key_values: list[str]
    ) -> Awaitable[list[Optional[Document]]]:
        """Async version of :meth:`get_by_key`."""
        ...
