"""
eval/run_eval.py
================
Measures the RETRIEVAL QUALITY and PERFORMANCE of the system against a labeled
eval set (eval/eval_set.json). This is how we back up the claim that retrieval is
"optimised in terms of accuracy and performance" with actual numbers instead of
assertions.

Metrics
-------
- Hit@1 / Hit@3 : did the correct source document appear in the top-1 / top-3?
- MRR           : Mean Reciprocal Rank — 1/rank of the first correct hit,
                  averaged over all questions. Rewards ranking the right doc high.
- Latency       : average wall-clock time per retrieval (ms).

It runs with whatever embedder is configured: real OpenAI embeddings if
OPENAI_API_KEY is set (use these numbers for your final slide), or the offline
FakeEmbedder otherwise (proves the harness works without a key).

Run:
    python -m eval.run_eval
"""

from __future__ import annotations
import json
import os
import time
from statistics import mean

from src import config
from src.ingest import build_vector_store
from src.llm import OpenAIEmbedder, FakeEmbedder


def load_cases():
    path = os.path.join(os.path.dirname(__file__), "eval_set.json")
    with open(path) as f:
        return json.load(f)["cases"]


def rank_of_correct(results, relevant_source):
    """1-based rank of the first chunk from the correct source, or None."""
    for i, (chunk, _score) in enumerate(results, start=1):
        if chunk.source == relevant_source:
            return i
    return None


def main():
    if os.environ.get("OPENAI_API_KEY"):
        embedder, mode = OpenAIEmbedder(), "OpenAI text-embedding-3-small (PRODUCTION)"
    else:
        embedder, mode = FakeEmbedder(), "FakeEmbedder (offline harness check)"

    print(f"Embedder: {mode}")
    store = build_vector_store(embedder)
    print(f"Indexed {len(store)} chunks\n")

    cases = load_cases()
    hits1 = hits3 = 0
    reciprocal_ranks = []
    latencies_ms = []

    print(f"{'#':>2}  {'Hit@1':>5}  {'Hit@3':>5}  {'rank':>4}  question")
    print("-" * 78)
    for i, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        q_vec = embedder.embed([case["question"]])[0]
        results = store.search(q_vec, top_k=config.TOP_K)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        rank = rank_of_correct(results, case["relevant_source"])
        h1 = rank == 1
        h3 = rank is not None and rank <= 3
        hits1 += int(h1)
        hits3 += int(h3)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        print(f"{i:>2}  {('Y' if h1 else '-'):>5}  {('Y' if h3 else '-'):>5}  "
              f"{(rank if rank else '-'):>4}  {case['question'][:48]}")

    n = len(cases)
    print("\n=== Retrieval metrics ===")
    print(f"Hit@1            : {hits1}/{n}  =  {hits1 / n:.0%}")
    print(f"Hit@3            : {hits3}/{n}  =  {hits3 / n:.0%}")
    print(f"MRR              : {mean(reciprocal_ranks):.3f}")
    print(f"Avg latency      : {mean(latencies_ms):.1f} ms/query")
    print(f"p95 latency      : {sorted(latencies_ms)[int(0.95 * n) - 1]:.1f} ms/query")
    print("\nNote: run with OPENAI_API_KEY set to record production-embedding "
          "numbers for your slide.")


if __name__ == "__main__":
    main()
