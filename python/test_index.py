"""Tests for IndexBuilder, Index, BuilderOptions, and search algorithms."""

import tempfile

import numpy as np
import pytest

import impact_index


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def build_test_index(tmpdir, rng, num_docs=100, num_terms=50, options=None, dtype=None):
    """Helper to build a small test index and return (index, dir)."""
    kwargs = {"options": options} if options else {}
    if dtype:
        kwargs["dtype"] = dtype
    builder = impact_index.IndexBuilder(tmpdir, **kwargs)
    is_int = dtype in ("int32", "int64")
    for doc_id in range(num_docs):
        n = rng.randint(5, min(20, num_terms))
        terms = rng.choice(num_terms, n, replace=False).astype(np.uint64)
        if is_int:
            values = rng.randint(1, 10, n).astype(np.float32)
        else:
            values = np.abs(rng.randn(n)).astype(np.float32) + 0.01
        builder.add(doc_id, terms, values)
    index = builder.build(True)
    return index


class TestBuilderOptions:
    def test_defaults(self):
        opts = impact_index.BuilderOptions()
        assert opts.checkpoint_frequency >= 0
        assert opts.in_memory_threshold >= 0

    def test_set_checkpoint_frequency(self):
        opts = impact_index.BuilderOptions()
        opts.checkpoint_frequency = 500
        assert opts.checkpoint_frequency == 500

    def test_set_in_memory_threshold(self):
        opts = impact_index.BuilderOptions()
        opts.in_memory_threshold = 1024
        assert opts.in_memory_threshold == 1024


class TestIndexBuilder:
    def test_build_and_search(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng)
            results = index.search_wand({1: 0.5, 5: 1.0}, 10)
            assert len(results) <= 10
            assert all(hasattr(r, "docid") and hasattr(r, "score") for r in results)

    def test_build_with_options(self, rng):
        opts = impact_index.BuilderOptions()
        opts.checkpoint_frequency = 50
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng, options=opts)
            results = index.search_wand({0: 1.0}, 5)
            assert isinstance(results, list)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    def test_build_different_dtypes(self, rng, dtype):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng, dtype=dtype)
            results = index.search_wand({0: 1.0}, 5)
            assert isinstance(results, list)

    def test_checkpoint(self, rng):
        with tempfile.TemporaryDirectory() as d:
            opts = impact_index.BuilderOptions()
            opts.checkpoint_frequency = 10
            builder = impact_index.IndexBuilder(d, opts)
            for doc_id in range(50):
                terms = np.array([0, 1, 2], dtype=np.uint64)
                values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                builder.add(doc_id, terms, values)
            cp = builder.get_checkpoint_doc_id()
            # With 50 docs and checkpoint_frequency=10, there should be a checkpoint
            assert cp is not None


class TestIndex:
    def test_load_from_disk(self, rng):
        with tempfile.TemporaryDirectory() as d:
            build_test_index(d, rng)
            loaded = impact_index.Index.load(d, True)
            results = loaded.search_wand({0: 1.0}, 5)
            assert isinstance(results, list)

    def test_num_postings(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng, num_terms=30)
            n = index.num_postings()
            assert n > 0
            assert n <= 30

    def test_postings_iterator(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng, num_terms=50)
            it = index.postings(0)
            assert it.length() > 0
            assert it.max_value() > 0
            assert it.max_doc_id() >= 0

            items = list(it)
            assert len(items) > 0
            assert all(hasattr(ti, "docid") and hasattr(ti, "value") for ti in items)

    def test_search_wand(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng)
            results = index.search_wand({0: 1.0, 1: 0.5}, 10)
            assert len(results) <= 10
            # Results should be sorted by score descending
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_maxscore(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng)
            results = index.search_maxscore({0: 1.0, 1: 0.5}, 10)
            assert len(results) <= 10
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_wand_maxscore_agree(self, rng):
        """WAND and MaxScore should return the same top-k results."""
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng, num_docs=200)
            query = {0: 1.0, 3: 0.8, 7: 0.3}
            wand = index.search_wand(query, 10)
            maxscore = index.search_maxscore(query, 10)
            wand_docs = {r.docid for r in wand}
            maxscore_docs = {r.docid for r in maxscore}
            assert wand_docs == maxscore_docs

    def test_search_deprecated(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng)
            results = index.search({0: 1.0}, 5)
            assert isinstance(results, list)

    def test_empty_query(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng)
            results = index.search_wand({}, 10)
            assert results == []


class TestAsyncSearch:
    @pytest.mark.asyncio
    async def test_aio_search_wand(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng)
            results = await index.aio_search_wand({0: 1.0, 1: 0.5}, 10)
            assert len(results) <= 10

    @pytest.mark.asyncio
    async def test_aio_search_maxscore(self, rng):
        with tempfile.TemporaryDirectory() as d:
            index = build_test_index(d, rng)
            results = await index.aio_search_maxscore({0: 1.0, 1: 0.5}, 10)
            assert len(results) <= 10
