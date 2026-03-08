# Impact Index for Information Retrieval

A Python/Rust library for efficient sparse retrieval from neural information retrieval systems. Built on Rust with PyO3 bindings for high performance.

Unlike traditional IR libraries, **impact-index** specifically targets neural IR models with floating-point impact scores (no quantization assumed) and does not implement frequency-based algorithms.

## Features

- **Index construction** with checkpointing for crash recovery
- **WAND and MaxScore** search algorithms for top-k retrieval
- **Block-based compression** with Elias-Fano doc IDs and quantized impacts
- **Posting list splitting** by quantile for term impact decomposition
- **BMP (Block-Max Pruning)** for fast approximate search ([SIGIR 2024](https://github.com/pisa-engine/BMP))
- **Document store** with zstd compression and key-based retrieval
- **Async support** for non-blocking search and document retrieval

## Installation

```bash
pip install maturin
maturin develop --release
```

## Quick Example

```python
import numpy as np
import impact_index

# Build an index
builder = impact_index.IndexBuilder("/path/to/index")
builder.add(0, np.array([1, 5, 10], dtype=np.uintp),
            np.array([0.5, 1.2, 0.8], dtype=np.float32))
index = builder.build(in_memory=True)

# Search
results = index.search_wand({5: 1.0, 10: 0.5}, top_k=10)
for doc in results:
    print(f"Document {doc.docid}: {doc.score}")
```

## Documentation

Full documentation including guides on compression, BMP search, and the document store is available at:

**https://experimaestro-ir-rust.readthedocs.io/en/latest/index.html**
