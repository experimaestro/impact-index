#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ir-datasets",
#     "torch",
#     "transformers",
#     "numpy",
#     "tqdm",
#     "impact-index",
# ]
# ///
"""
SPLADE Indexing and Search Benchmark

This example demonstrates:
1. Loading a SPLADE model from HuggingFace
2. Encoding documents and queries using SPLADE
3. Building both impact-index and BMP indices
4. Searching with both index types
5. Measuring performance (CPU time, wall time, index size)
6. Computing IR metrics (MRR, NDCG, Recall)

Usage:
    uv run examples/splade_benchmark.py [--dataset msmarco-passage/dev/small] [--model naver/splade-cocondenser-ensembledistil]

Or with pip-installed dependencies:
    python examples/splade_benchmark.py [options]
"""

import argparse
import json
import resource
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Import the index libraries
import impact_index
from impact_index import BmpSearcher


def get_best_device() -> str:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def get_file_size_mb(path: Path) -> float:
    """Get total size of files in a directory in MB."""
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1024 / 1024


class SpladeEncoder:
    """SPLADE sparse encoder using HuggingFace transformers."""

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", device: str = None):
        self.device = device or get_best_device()
        print(f"Loading SPLADE model '{model_name}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Check for multi-GPU setup
        self.multi_gpu = False
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            print(f"  Using DataParallel with {num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model)
            self.multi_gpu = True

        self.model.to(self.device)
        self.model.eval()

        # Get vocabulary for term mapping
        self.vocab = self.tokenizer.get_vocab()
        self.id2token = {v: k for k, v in self.vocab.items()}

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 256) -> List[Dict[int, float]]:
        """
        Encode texts into sparse SPLADE representations.

        Returns list of dicts mapping term_id -> weight.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits

        # SPLADE aggregation: max over sequence, then ReLU + log1p
        # Shape: (batch, seq_len, vocab_size) -> (batch, vocab_size)
        weights = torch.max(
            torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        ).values

        results = []
        for i in range(weights.shape[0]):
            sparse_vec = {}
            nonzero = weights[i].nonzero(as_tuple=True)[0]
            for idx in nonzero:
                term_id = idx.item()
                weight = weights[i, term_id].item()
                if weight > 0:
                    sparse_vec[term_id] = weight
            results.append(sparse_vec)

        return results

    def encode_batch(
        self, texts: List[str], batch_size: int = 32, max_length: int = 256, desc: str = "Encoding", unit: str = "item"
    ) -> List[Dict[int, float]]:
        """Encode texts in batches with progress bar (shows individual items, not batches)."""
        results = []
        with tqdm(total=len(texts), desc=desc, unit=unit) as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_results = self.encode(batch, max_length)
                results.extend(batch_results)
                pbar.update(len(batch))
        return results


def load_queries_and_qrels(dataset_name: str, max_queries: int = None):
    """Load queries and qrels from dataset (lightweight, kept in memory)."""
    import ir_datasets

    print(f"Loading dataset '{dataset_name}'...")
    dataset = ir_datasets.load(dataset_name)

    # Load queries
    queries = []
    query_ids = []
    print("Loading queries...")
    for i, query in enumerate(dataset.queries_iter()):
        if max_queries > 0 and i >= max_queries:
            break
        queries.append(query.text if hasattr(query, "text") else str(query))
        query_ids.append(query.query_id)

    # Load qrels
    qrels = defaultdict(dict)
    print("Loading qrels...")
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    return dataset, queries, query_ids, qrels


def doc_id_mapping_path(index_dir: Path) -> Path:
    """Path to store doc_id mapping for evaluation."""
    return index_dir / "doc_ids.json"


def build_impact_index_streaming(
    encoder: SpladeEncoder,
    dataset,
    index_dir: Path,
    batch_size: int = 32,
    max_docs: int = None,
) -> Tuple[impact_index.Index, List[str]]:
    """Build impact-index from documents using streaming (no full doc list in memory)."""
    print("\n=== Building Impact Index (Streaming) ===")

    done_file = index_dir / ".done"
    doc_ids_file = doc_id_mapping_path(index_dir)

    # Check if index already exists
    if done_file.exists() and doc_ids_file.exists():
        print("  Index already built, loading from disk...")
        index = impact_index.Index.load(str(index_dir), in_memory=True)
        with open(doc_ids_file, "r") as f:
            doc_ids = json.load(f)
        print(f"  Index size: {get_file_size_mb(index_dir):.2f} MB")
        print(f"  Documents indexed: {len(doc_ids)}")
        return index, doc_ids

    options = impact_index.BuilderOptions()
    options.checkpoint_frequency = 10000
    indexer = impact_index.IndexBuilder(str(index_dir), options)

    # Check for checkpoint to resume from
    checkpoint_doc_id = indexer.get_checkpoint_doc_id()
    resume_from = checkpoint_doc_id + 1 if checkpoint_doc_id is not None else 0
    if resume_from > 0:
        print(f"  Resuming from checkpoint at doc {resume_from}")

    start_time = time.time()
    start_cpu = time.process_time()

    # Stream documents in batches
    doc_ids = []
    batch_texts = []
    batch_doc_ids = []
    current_doc_idx = 0  # Actual doc index in the dataset

    # Get total document count from ir_datasets if available
    total_docs = None
    if hasattr(dataset, 'docs_count'):
        total_docs = dataset.docs_count()
        if max_docs > 0:
            total_docs = min(total_docs, max_docs)
    elif max_docs > 0:
        total_docs = max_docs

    print("  Streaming documents...")
    with tqdm(total=total_docs, desc="Indexing", unit="doc", initial=resume_from) as pbar:
        for doc in dataset.docs_iter():
            if max_docs > 0 and current_doc_idx >= max_docs:
                break

            doc_id = doc.doc_id
            doc_ids.append(doc_id)

            # Skip encoding for documents before resume point (but still collect doc_ids)
            if current_doc_idx < resume_from:
                current_doc_idx += 1
                continue

            doc_text = doc.text if hasattr(doc, "text") else doc.body
            batch_texts.append(doc_text)
            batch_doc_ids.append(doc_id)

            # Process batch when full
            if len(batch_texts) >= batch_size:
                encodings = encoder.encode(batch_texts)
                for j, sparse_vec in enumerate(encodings):
                    doc_idx = current_doc_idx - len(batch_texts) + 1 + j
                    if sparse_vec:
                        terms = np.array(list(sparse_vec.keys()), dtype=np.uint64)
                        values = np.array(list(sparse_vec.values()), dtype=np.float32)
                        indexer.add(doc_idx, terms, values)
                pbar.update(len(batch_texts))
                batch_texts = []
                batch_doc_ids = []

            current_doc_idx += 1

        # Process remaining documents in the last batch
        if batch_texts:
            encodings = encoder.encode(batch_texts)
            for j, sparse_vec in enumerate(encodings):
                doc_idx = current_doc_idx - len(batch_texts) + j
                if sparse_vec:
                    terms = np.array(list(sparse_vec.keys()), dtype=np.uint64)
                    values = np.array(list(sparse_vec.values()), dtype=np.float32)
                    indexer.add(doc_idx, terms, values)
            pbar.update(len(batch_texts))

    # Build the index
    index = indexer.build(in_memory=True)

    end_time = time.time()
    end_cpu = time.process_time()

    # Save doc_ids mapping for evaluation
    with open(doc_ids_file, "w") as f:
        json.dump(doc_ids, f)

    # Mark as done
    done_file.touch()

    print(f"  Documents indexed: {len(doc_ids)}")
    print(f"  Wall time: {end_time - start_time:.2f}s")
    print(f"  CPU time: {end_cpu - start_cpu:.2f}s")
    print(f"  Index size: {get_file_size_mb(index_dir):.2f} MB")

    return index, doc_ids


def build_bmp_index(
    index: impact_index.Index,
    bmp_path: Path,
    bsize: int = 64,
    compress_range: bool = True,
    use_streaming: bool = True,
) -> BmpSearcher:
    """Convert impact-index to BMP format."""
    print("\n=== Building BMP Index ===")
    method = "streaming" if use_streaming else "legacy"
    print(f"  Method: {method}")

    done_file = bmp_path.with_suffix(".done")

    # Check if BMP index already exists
    if done_file.exists() and bmp_path.exists():
        print("  BMP index already built, loading from disk...")
        print(f"  BMP index size: {get_file_size_mb(bmp_path):.2f} MB")
        return BmpSearcher(str(bmp_path))

    start_time = time.time()
    start_cpu = time.process_time()
    start_mem = get_memory_usage_mb()

    if use_streaming:
        index.to_bmp_streaming(str(bmp_path), bsize, compress_range)
    else:
        index.to_bmp(str(bmp_path), bsize, compress_range)

    end_time = time.time()
    end_cpu = time.process_time()
    end_mem = get_memory_usage_mb()

    # Mark as done
    done_file.touch()

    print(f"  Wall time: {end_time - start_time:.2f}s")
    print(f"  CPU time: {end_cpu - start_cpu:.2f}s")
    print(f"  Memory delta: {end_mem - start_mem:.2f} MB")
    print(f"  BMP index size: {get_file_size_mb(bmp_path):.2f} MB")

    return BmpSearcher(str(bmp_path))


def search_impact_index(
    index: impact_index.Index,
    query_encodings: List[Dict[int, float]],
    top_k: int = 100,
    method: str = "wand",
) -> Tuple[List[List[Tuple[int, float]]], float, float]:
    """Search using impact-index with WAND or MaxScore algorithm."""
    results = []

    start_time = time.time()
    start_cpu = time.process_time()

    search_fn = index.search_wand if method == "wand" else index.search_maxscore
    desc = f"Searching ({method.upper()})"

    for sparse_vec in tqdm(query_encodings, desc=desc, unit="query"):
        if sparse_vec:
            hits = search_fn(sparse_vec, top_k)
            results.append([(h.docid, h.score) for h in hits])
        else:
            results.append([])

    end_time = time.time()
    end_cpu = time.process_time()

    wall_time = end_time - start_time
    cpu_time = end_cpu - start_cpu

    return results, wall_time, cpu_time


def search_bmp_index(
    searcher: BmpSearcher,
    query_encodings: List[Dict[int, float]],
    top_k: int = 100,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[List[List[Tuple[int, float]]], float, float]:
    """Search using BMP index."""
    results = []

    start_time = time.time()
    start_cpu = time.process_time()

    for sparse_vec in tqdm(query_encodings, desc="Searching (BMP)"):
        if sparse_vec:
            # BMP expects string keys
            str_query = {str(k): v for k, v in sparse_vec.items()}
            doc_ids, scores = searcher.search(str_query, k=top_k, alpha=alpha, beta=beta)
            # Convert string doc_ids back to int
            results.append([(int(d), s) for d, s in zip(doc_ids, scores)])
        else:
            results.append([])

    end_time = time.time()
    end_cpu = time.process_time()

    wall_time = end_time - start_time
    cpu_time = end_cpu - start_cpu

    return results, wall_time, cpu_time


def compute_metrics(
    results: List[List[Tuple[int, float]]],
    query_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
    doc_ids: List[str],
    k_values: List[int] = [10, 100],
) -> Dict[str, float]:
    """Compute IR metrics: MRR, NDCG, Recall."""
    metrics = {}

    for k in k_values:
        mrr_sum = 0.0
        ndcg_sum = 0.0
        recall_sum = 0.0
        num_queries = 0

        for i, (query_id, hits) in enumerate(zip(query_ids, results)):
            if query_id not in qrels:
                continue

            rel_docs = qrels[query_id]
            num_relevant = sum(1 for r in rel_docs.values() if r > 0)

            if num_relevant == 0:
                continue

            num_queries += 1

            # Get retrieved doc_ids
            retrieved = [doc_ids[doc_idx] for doc_idx, _ in hits[:k]]

            # MRR
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in rel_docs and rel_docs[doc_id] > 0:
                    mrr_sum += 1.0 / rank
                    break

            # Recall@k
            num_retrieved_relevant = sum(1 for d in retrieved if d in rel_docs and rel_docs[d] > 0)
            recall_sum += num_retrieved_relevant / min(num_relevant, k)

            # NDCG@k
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in rel_docs:
                    rel = rel_docs[doc_id]
                    dcg += (2**rel - 1) / np.log2(rank + 1)

            # Ideal DCG
            ideal_rels = sorted([r for r in rel_docs.values() if r > 0], reverse=True)[:k]
            idcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))

            if idcg > 0:
                ndcg_sum += dcg / idcg

        if num_queries > 0:
            metrics[f"MRR@{k}"] = mrr_sum / num_queries
            metrics[f"NDCG@{k}"] = ndcg_sum / num_queries
            metrics[f"Recall@{k}"] = recall_sum / num_queries

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SPLADE Indexing and Search Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco-passage/dev/small",
        help="ir_datasets dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="naver/splade_v2_max",
        help="HuggingFace SPLADE model name",
    )
    parser.add_argument("--max-docs", type=int, default=0, help="Maximum number of documents to index (0 = all)")
    parser.add_argument("--max-queries", type=int, default=0, help="Maximum number of queries to evaluate (0 = all)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--top-k", type=int, default=100, help="Number of results to retrieve")
    parser.add_argument("--bsize", type=int, default=64, help="BMP block size")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: temp)")
    parser.add_argument(
        "--compare-bmp-methods",
        action="store_true",
        help="Compare legacy vs streaming BMP conversion",
    )
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)
        cleanup = True

    try:
        # Initialize encoder first (needed for both queries and documents)
        encoder = SpladeEncoder(args.model)

        # Load dataset for streaming
        import ir_datasets
        print(f"Loading dataset '{args.dataset}'...")
        dataset = ir_datasets.load(args.dataset)

        # Build impact-index using streaming (documents not loaded into memory)
        # Do this first since it's the most time-consuming step
        index_dir = output_dir / "impact_index"
        index_dir.mkdir(exist_ok=True)
        impact_idx, doc_ids = build_impact_index_streaming(
            encoder, dataset, index_dir, batch_size=args.batch_size, max_docs=args.max_docs
        )

        # Load queries and qrels (small, kept in memory)
        dataset, queries, query_ids, qrels = load_queries_and_qrels(
            args.dataset, max_queries=args.max_queries
        )
        print(f"\nLoaded {len(queries)} queries")

        # Encode queries
        print("\n=== Encoding Queries ===")
        query_encodings = encoder.encode_batch(
            queries, batch_size=args.batch_size, desc="Encoding queries", unit="query"
        )

        # Build BMP index (streaming)
        bmp_path = output_dir / "index_streaming.bmp"
        bmp_searcher = build_bmp_index(impact_idx, bmp_path, bsize=args.bsize, use_streaming=True)

        # Optionally compare with legacy BMP conversion
        if args.compare_bmp_methods:
            bmp_legacy_path = output_dir / "index_legacy.bmp"
            bmp_legacy_searcher = build_bmp_index(
                impact_idx, bmp_legacy_path, bsize=args.bsize, use_streaming=False
            )

        # Search with WAND
        print("\n=== Searching with WAND ===")
        wand_results, wand_wall, wand_cpu = search_impact_index(
            impact_idx, query_encodings, top_k=args.top_k, method="wand"
        )
        print(f"  Wall time: {wand_wall:.2f}s ({len(queries)/wand_wall:.1f} q/s)")
        print(f"  CPU time: {wand_cpu:.2f}s")

        # Search with MaxScore
        print("\n=== Searching with MaxScore ===")
        maxscore_results, maxscore_wall, maxscore_cpu = search_impact_index(
            impact_idx, query_encodings, top_k=args.top_k, method="maxscore"
        )
        print(f"  Wall time: {maxscore_wall:.2f}s ({len(queries)/maxscore_wall:.1f} q/s)")
        print(f"  CPU time: {maxscore_cpu:.2f}s")

        # Search with BMP
        print("\n=== Searching with BMP ===")
        bmp_results, bmp_wall, bmp_cpu = search_bmp_index(bmp_searcher, query_encodings, top_k=args.top_k)
        print(f"  Wall time: {bmp_wall:.2f}s ({len(queries)/bmp_wall:.1f} q/s)")
        print(f"  CPU time: {bmp_cpu:.2f}s")

        # Compute metrics
        print("\n=== IR Metrics ===")
        print("\nWAND:")
        wand_metrics = compute_metrics(wand_results, query_ids, qrels, doc_ids)
        for metric, value in sorted(wand_metrics.items()):
            print(f"  {metric}: {value:.4f}")

        print("\nMaxScore:")
        maxscore_metrics = compute_metrics(maxscore_results, query_ids, qrels, doc_ids)
        for metric, value in sorted(maxscore_metrics.items()):
            print(f"  {metric}: {value:.4f}")

        print("\nBMP:")
        bmp_metrics = compute_metrics(bmp_results, query_ids, qrels, doc_ids)
        for metric, value in sorted(bmp_metrics.items()):
            print(f"  {metric}: {value:.4f}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\nDataset: {args.dataset}")
        print(f"Documents: {len(doc_ids)}")
        print(f"Queries: {len(queries)}")
        print(f"Model: {args.model}")

        print("\nIndex Sizes:")
        print(f"  Standard Index: {get_file_size_mb(index_dir):.2f} MB")
        print(f"  BMP Index: {get_file_size_mb(bmp_path):.2f} MB")

        print("\nSearch Performance:")
        print(f"  WAND:     {wand_wall:.2f}s wall, {wand_cpu:.2f}s CPU ({len(queries)/wand_wall:.1f} q/s)")
        print(f"  MaxScore: {maxscore_wall:.2f}s wall, {maxscore_cpu:.2f}s CPU ({len(queries)/maxscore_wall:.1f} q/s)")
        print(f"  BMP:      {bmp_wall:.2f}s wall, {bmp_cpu:.2f}s CPU ({len(queries)/bmp_wall:.1f} q/s)")

        print("\nKey Metrics:")
        for k in [10, 100]:
            if f"MRR@{k}" in wand_metrics:
                print(f"  MRR@{k}:    WAND={wand_metrics[f'MRR@{k}']:.4f}, MaxScore={maxscore_metrics[f'MRR@{k}']:.4f}, BMP={bmp_metrics[f'MRR@{k}']:.4f}")
            if f"NDCG@{k}" in wand_metrics:
                print(f"  NDCG@{k}:   WAND={wand_metrics[f'NDCG@{k}']:.4f}, MaxScore={maxscore_metrics[f'NDCG@{k}']:.4f}, BMP={bmp_metrics[f'NDCG@{k}']:.4f}")
            if f"Recall@{k}" in wand_metrics:
                print(f"  Recall@{k}: WAND={wand_metrics[f'Recall@{k}']:.4f}, MaxScore={maxscore_metrics[f'Recall@{k}']:.4f}, BMP={bmp_metrics[f'Recall@{k}']:.4f}")

        # Save results to JSON
        results_path = output_dir / "benchmark_results.json"
        results = {
            "config": vars(args),
            "dataset_stats": {
                "num_docs": len(doc_ids),
                "num_queries": len(queries),
            },
            "index_sizes_mb": {
                "standard_index": get_file_size_mb(index_dir),
                "bmp_index": get_file_size_mb(bmp_path),
            },
            "search_time_seconds": {
                "wand": {"wall": wand_wall, "cpu": wand_cpu},
                "maxscore": {"wall": maxscore_wall, "cpu": maxscore_cpu},
                "bmp": {"wall": bmp_wall, "cpu": bmp_cpu},
            },
            "metrics": {
                "wand": wand_metrics,
                "maxscore": maxscore_metrics,
                "bmp": bmp_metrics,
            },
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    finally:
        if cleanup:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
