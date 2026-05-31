"""
Tests for retrieval quality. We check that a topical query retrieves the
chunk from the matching document, using the FakeEmbedder (word-overlap based),
so these run offline and deterministically.
"""

from src.ingest import build_vector_store
from src.vectorstore import cosine_similarity
from src.llm import FakeEmbedder


def test_cosine_similarity_bounds():
    """Identical vectors -> 1.0; orthogonal -> 0.0."""
    assert abs(cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9
    assert abs(cosine_similarity([1, 0, 0], [0, 1, 0]) - 0.0) < 1e-9


def test_storage_query_retrieves_storage_doc():
    """A question about SSD controllers should surface the NAND storage doc."""
    embedder = FakeEmbedder()
    store = build_vector_store(embedder)
    query_vec = embedder.embed(["what does an SSD flash controller do"])[0]
    results = store.search(query_vec, top_k=3)
    sources = [chunk.source for chunk, _ in results]
    assert any("nand" in s.lower() or "storage" in s.lower() for s in sources)


def test_finetuning_query_retrieves_finetuning_doc():
    """A question about LoRA should surface the fine-tuning doc."""
    embedder = FakeEmbedder()
    store = build_vector_store(embedder)
    query_vec = embedder.embed(["LoRA parameter efficient fine-tuning memory"])[0]
    results = store.search(query_vec, top_k=3)
    sources = [chunk.source for chunk, _ in results]
    assert any("finetuning" in s.lower() or "tuning" in s.lower() for s in sources)


def test_results_sorted_by_score():
    """search() must return chunks in descending similarity order."""
    embedder = FakeEmbedder()
    store = build_vector_store(embedder)
    query_vec = embedder.embed(["vector embeddings cosine similarity"])[0]
    results = store.search(query_vec, top_k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
