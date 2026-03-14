"""Tests for compression and split transforms."""

import numpy as np
import pytest

import impact_index


@pytest.fixture
def rng():
    return np.random.RandomState(123)


def build_raw_index(tmpdir, rng, num_docs=100, num_terms=30):
    builder = impact_index.IndexBuilder(tmpdir)
    for doc_id in range(num_docs):
        n = rng.randint(5, 15)
        terms = rng.choice(num_terms, n, replace=False).astype(np.uint64)
        values = np.abs(rng.randn(n)).astype(np.float32) + 0.01
        builder.add(doc_id, terms, values)
    return builder.build(True)


class TestCompressionTransform:
    def test_compress_and_search(self, rng, tmp_path):
        raw_dir = str(tmp_path / "raw")
        compressed_dir = str(tmp_path / "compressed")
        import os

        os.makedirs(raw_dir)

        index = build_raw_index(raw_dir, rng)

        ef = impact_index.EliasFanoCompressor()
        iq = impact_index.GlobalImpactQuantizer(8)
        transform = impact_index.CompressionTransform(128, ef, iq)
        transform.process(compressed_dir, index)

        compressed = impact_index.Index.load(compressed_dir, True)
        results = compressed.search_wand({0: 1.0, 1: 0.5}, 10)
        assert len(results) <= 10
        assert all(r.score > 0 for r in results)

    def test_fixed_range_quantizer(self, rng, tmp_path):
        raw_dir = str(tmp_path / "raw")
        compressed_dir = str(tmp_path / "compressed")
        import os

        os.makedirs(raw_dir)

        index = build_raw_index(raw_dir, rng)

        ef = impact_index.EliasFanoCompressor()
        iq = impact_index.ImpactQuantizer(8, 0.0, 5.0)
        transform = impact_index.CompressionTransform(64, ef, iq)
        transform.process(compressed_dir, index)

        compressed = impact_index.Index.load(compressed_dir, True)
        results = compressed.search_wand({0: 1.0}, 5)
        assert isinstance(results, list)


class TestSplitIndexTransform:
    def test_split_and_search(self, rng, tmp_path):
        raw_dir = str(tmp_path / "raw")
        split_dir = str(tmp_path / "split")
        import os

        os.makedirs(raw_dir)

        index = build_raw_index(raw_dir, rng)

        ef = impact_index.EliasFanoCompressor()
        iq = impact_index.GlobalImpactQuantizer(8)
        compress = impact_index.CompressionTransform(128, ef, iq)
        split = impact_index.SplitIndexTransform([0.9], compress)
        split.process(split_dir, index)

        split_index = impact_index.Index.load(split_dir, True)
        results = split_index.search_wand({0: 1.0, 1: 0.5}, 10)
        assert len(results) <= 10
