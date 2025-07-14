import numpy as np
import impact_index
import tempfile

NUM_DOCS = 1_000
NUM_TERMS = 100

np.random.seed(10)

# Sparse indexer
with tempfile.TemporaryDirectory() as dir:
    print(f"Using directory {dir}")
    options = impact_index.BuilderOptions()
    options.checkpoint_frequency = 600
    indexer = impact_index.IndexBuilder(dir, options)

    for doc_id in range(NUM_DOCS):
        num_terms = np.random.randint(10, 50)
        terms = np.random.choice(range(NUM_TERMS), num_terms, replace=False).astype(np.uint64)
        impacts = np.abs(np.random.randn(num_terms)).astype(np.float32)
        indexer.add(doc_id, terms, impacts)

    index = indexer.build(True)

    # for i in range(max(50, NUM_TERMS)):
    #     for t in indexer.iter(i):
    #         print(i, t.docid, t.value)
    q = {1: .2, 5: .52, 12: .1}

    print([(d.docid, d.score) for d in index.search(q, 10)])

    print("Load index from disk")
    index = impact_index.impact_index.Index.load(dir, True)
    print([(d.docid, d.score) for d in index.search(q, 10)])
