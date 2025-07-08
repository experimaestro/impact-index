from pathlib import Path
from tempfile import tempdir
from bmp import Searcher
import numpy as np
import impact_index
import tempfile

NUM_DOCS = 1_000
NUM_TERMS = 100
np.random.seed(10)

# Sparse indexer
with tempfile.TemporaryDirectory() as dir:
    print(f"Using directory {dir}")
    indexer = impact_index.IndexBuilder(dir)

    for doc_id in range(NUM_DOCS):
        num_terms = np.random.randint(10, 50)
        terms = np.random.choice(range(NUM_TERMS), num_terms, replace=False).astype(np.uint64)
        impacts = np.abs(np.random.randn(num_terms)).astype(np.float32)
        indexer.add(doc_id, terms, impacts)

    # Convert into BMP
    index = indexer.build(True)
    bmp_index_path = str(Path(dir) / "index.bmp")
    index.to_bmp(bmp_index_path, 32, True)

    searcher = Searcher(str(bmp_index_path))
    print(searcher.search({"1": .2, "5": .52, "12": .1}, k=10, alpha=1, beta=1))