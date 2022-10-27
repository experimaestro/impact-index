from tempfile import tempdir
import numpy as np
import xpmir_rust
import tempfile

NUM_DOCS = 1_000
NUM_TERMS = 100

# Sparse indexer
with tempfile.TemporaryDirectory() as dir:
    print(f"Using directory {dir}")
    indexer = xpmir_rust.index.SparseIndexer(dir)

    for doc_id in range(NUM_DOCS):
        num_terms = np.random.randint(10, 50)
        terms = np.random.choice(range(NUM_TERMS), num_terms).astype(np.uint64)
        impacts = np.random.randn((num_terms)).astype(np.float32)
        indexer.add(doc_id, terms, impacts)


    index = indexer.build()

    # for i in range(max(50, NUM_TERMS)):
    #     for t in indexer.iter(i):
    #         print(i, t.docid, t.value)

    print([(d.docid, d.score) for d in index.search({1: .2, 5: .5}, 10)])