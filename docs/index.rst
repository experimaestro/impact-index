impact-index
=============

A Python/Rust library for efficient sparse retrieval from neural information
retrieval systems. Built on Rust with PyO3 bindings for high performance.

Unlike traditional IR libraries, **impact-index** specifically targets neural
IR models with floating-point impact scores (no quantization assumed) and does
not implement frequency-based algorithms.

Features
--------

- **Index construction** with checkpointing for crash recovery
- **WAND and MaxScore** search algorithms for top-k retrieval
- **Block-based compression** with Elias-Fano doc IDs and quantized impacts
- **Posting list splitting** by quantile for term impact decomposition
- **BMP (Block-Max Pruning)** integration for fast approximate search
- **Document store** with zstd compression and key-based retrieval
- **Async support** for non-blocking search and document retrieval

Quick Start
-----------

Building an index::

    import numpy as np
    import impact_index

    builder = impact_index.IndexBuilder("/path/to/index")
    terms = np.array([0, 5, 42], dtype=np.uintp)
    values = np.array([1.2, 0.5, 3.1], dtype=np.float32)
    builder.add(0, terms, values)
    index = builder.build(in_memory=True)

Searching::

    results = index.search_wand({42: 1.5, 100: 0.8}, top_k=10)
    for doc in results:
        print(doc.docid, doc.score)

Compressing an index::

    transform = impact_index.CompressionTransform(
        max_block_size=128,
        doc_ids_compressor=impact_index.EliasFanoCompressor(),
        impacts_compressor=impact_index.GlobalImpactQuantizer(nbits=8),
    )
    transform.process("/path/to/compressed", index)

    # Load the compressed index
    compressed = impact_index.Index.load("/path/to/compressed", in_memory=True)

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Installation
------------

Requires Rust toolchain and Python >= 3.8::

    pip install maturin
    maturin develop --release
