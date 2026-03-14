"""Tests for converting an index to BMP format."""

import numpy as np
import pytest

import impact_index


@pytest.fixture
def rng():
    return np.random.RandomState(77)


def build_raw_index(tmpdir, rng, num_docs=100, num_terms=30):
    builder = impact_index.IndexBuilder(tmpdir)
    for doc_id in range(num_docs):
        n = rng.randint(5, 15)
        terms = rng.choice(num_terms, n, replace=False).astype(np.uint64)
        values = np.abs(rng.randn(n)).astype(np.float32) + 0.01
        builder.add(doc_id, terms, values)
    return builder.build(True)


class TestBmpConversion:
    def test_to_bmp(self, rng, tmp_path):
        raw_dir = str(tmp_path / "raw")
        bmp_dir = str(tmp_path / "bmp")
        import os

        os.makedirs(raw_dir)

        index = build_raw_index(raw_dir, rng)
        index.to_bmp(bmp_dir, bsize=64, compress_range=False)

        searcher = impact_index.BmpSearcher(bmp_dir)
        assert searcher.num_documents() == 100

    def test_to_bmp_streaming(self, rng, tmp_path):
        raw_dir = str(tmp_path / "raw")
        bmp_dir = str(tmp_path / "bmp")
        import os

        os.makedirs(raw_dir)

        index = build_raw_index(raw_dir, rng)
        index.to_bmp_streaming(bmp_dir, bsize=64, compress_range=False)

        searcher = impact_index.BmpSearcher(bmp_dir)
        assert searcher.num_documents() == 100
