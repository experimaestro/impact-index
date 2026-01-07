# Impact Index for Information Retrieval

This package is a library that implements efficient algorithms for sparse representations from neural information retrieval systems. Contrarily to other libraries, this one specifically targets neural IR models, and does not suppose any quantization. It also does not implement "standard" IR algorithms based on term frequencies.

## Installation

```bash
pip install maturin
maturin develop --release
```

## Python API

The library exposes the `impact_index` Python module for building and searching sparse indices.

### Basic Usage

```python
import numpy as np
from impact_index import IndexBuilder, Index

# Build an index
builder = IndexBuilder("/path/to/index")
# Add documents: docid, term_indices (numpy array), impact_values (numpy array)
builder.add(0, np.array([1, 5, 10]), np.array([0.5, 1.2, 0.8]))
builder.add(1, np.array([2, 5, 8]), np.array([0.3, 0.9, 1.1]))
index = builder.build(in_memory=True)

# Search
query = {5: 1.0, 10: 0.5}  # {term_index: weight}
results = index.search_wand(query, top_k=10)
for doc in results:
    print(f"Document {doc.docid}: {doc.score}")
```

### Classes

#### `BuilderOptions`

Options for index construction.

```python
from impact_index import BuilderOptions

options = BuilderOptions()
options.checkpoint_frequency = 100000  # Checkpoint every N documents
options.in_memory_threshold = 1000000  # In-memory threshold
```

#### `IndexBuilder`

Builds a sparse index from documents.

```python
IndexBuilder(folder: str, options: BuilderOptions = None)
```

**Methods:**
- `add(docid: int, terms: np.ndarray[int], values: np.ndarray[float])` - Add a document with term indices and impact values
- `get_checkpoint_doc_id() -> int | None` - Get the last checkpointed document ID (useful for resuming indexing)
- `build(in_memory: bool) -> Index` - Finalize and return the index

#### `Index`

A sparse index supporting efficient top-k retrieval.

```python
Index.load(folder: str, in_memory: bool) -> Index
```

**Search Methods:**
- `search_wand(query: dict, top_k: int) -> list[ScoredDocument]` - WAND algorithm
- `search_maxscore(query: dict, top_k: int) -> list[ScoredDocument]` - MaxScore algorithm
- `aio_search_wand(query: dict, top_k: int) -> Coroutine` - Async WAND search
- `aio_search_maxscore(query: dict, top_k: int) -> Coroutine` - Async MaxScore search

**Other Methods:**
- `postings(term: int) -> SparseIndexIterator` - Get posting list iterator for a term
- `num_postings() -> int` - Total number of posting lists
- `to_bmp(output: str, bsize: int, compress_range: bool)` - Convert to BMP format
- `to_bmp_streaming(output: str, bsize: int, compress_range: bool)` - Memory-efficient BMP conversion

#### `SparseIndexIterator`

Iterator over a term's posting list.

```python
iterator = index.postings(term_id)
print(f"Length: {iterator.length()}")
print(f"Max impact: {iterator.max_value()}")
print(f"Max doc ID: {iterator.max_doc_id()}")

for posting in iterator:
    print(f"Doc {posting.docid}: {posting.value}")
```

#### `ScoredDocument`

Search result with document ID and score.

- `docid: int` - Document identifier
- `score: float` - Retrieval score

### Compression and Transforms

Apply compression to reduce index size:

```python
from impact_index import (
    Index, CompressionTransform,
    EliasFanoCompressor, ImpactQuantizer, GlobalImpactQuantizer
)

index = Index.load("/path/to/raw_index", in_memory=True)

# Create compressors
docid_compressor = EliasFanoCompressor()
impact_compressor = ImpactQuantizer(nbits=8, min=0.0, max=10.0)
# Or use global quantization:
# impact_compressor = GlobalImpactQuantizer(nbits=8)

# Apply compression
transform = CompressionTransform(
    max_block_size=128,
    doc_ids_compressor=docid_compressor,
    impacts_compressor=impact_compressor
)
transform.process("/path/to/compressed_index", index)
```

#### `SplitIndexTransform`

Split index by impact quantiles for tiered retrieval:

```python
from impact_index import SplitIndexTransform, CompressionTransform

base_transform = CompressionTransform(128, docid_compressor, impact_compressor)
split_transform = SplitIndexTransform(
    quantiles=[0.5, 0.9],  # Split at 50th and 90th percentile
    sink=base_transform
)
split_transform.process("/path/to/split_index", index)
```

### BMP (Block-Max Pruning) Search

Fast approximate search using the BMP algorithm from the [BMP repository](https://github.com/pisa-engine/BMP), which implements "Faster Learned Sparse Retrieval with Block-Max Pruning" (SIGIR 2024).

```python
from impact_index import Index, BmpSearcher

# Convert an existing index to BMP format
index = Index.load("/path/to/index", in_memory=True)
index.to_bmp_streaming("/path/to/bmp_index.bin", bsize=64, compress_range=True)

# Load and search with BMP
searcher = BmpSearcher("/path/to/bmp_index.bin")
print(f"Documents: {searcher.num_documents()}")

# Query uses string term IDs
query = {"term1": 1.0, "term2": 0.5}
doc_ids, scores = searcher.search(query, k=10, alpha=1.0, beta=1.0)
for docid, score in zip(doc_ids, scores):
    print(f"{docid}: {score}")
```

**BMP Conversion Methods:**
- `to_bmp_streaming(output, bsize, compress_range)` - Recommended. Memory-efficient streaming conversion using O(num_terms × num_blocks) memory
- `to_bmp(output, bsize, compress_range)` - Legacy method using O(total_postings) memory

**BMP Search Parameters:**
- `k` - Number of results to return
- `alpha` - Controls early termination aggressiveness (default: 1.0)
- `beta` - Controls block skipping (default: 1.0)

