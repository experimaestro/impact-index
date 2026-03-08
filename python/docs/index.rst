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

Installation
------------

Requires Rust toolchain and Python >= 3.8::

    pip install maturin
    maturin develop --release

.. toctree::
   :maxdepth: 2
   :caption: Contents

   guide
   api
