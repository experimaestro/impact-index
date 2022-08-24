from tempfile import tempdir
import numpy as np
import xpmir_rust
import tempfile

# Sparse indexer
with tempfile.TemporaryDirectory() as dir:
    indexer = xpmir_rust.index.SparseIndexer(dir)
    terms = np.array([0, 1, 2, 8], dtype=np.uint64)
    impacts = np.array([.2, .3, 1.2, .2], dtype=np.float32)
    indexer.add(0, terms, impacts)
