"""
Tests for the ingestion pipeline: chunking and vector-store building.
These need no API key — they use the FakeEmbedder.
"""

from src.ingest import chunk_text, build_vector_store
from src.llm import FakeEmbedder


def test_chunking_respects_size():
    """Chunks should not be wildly larger than the target size."""
    text = "\n\n".join(f"Paragraph number {i} with some words." for i in range(50))
    chunks = chunk_text(text, size=200, overlap=20)
    assert len(chunks) > 1                      # long text must split
    assert all(len(c) <= 400 for c in chunks)   # generous upper bound w/ overlap


def test_chunking_keeps_all_content():
    """No paragraph should be silently dropped during chunking."""
    text = "\n\n".join(["Alpha content here.", "Bravo content here.", "Charlie content."])
    chunks = chunk_text(text, size=50, overlap=10)
    joined = " ".join(chunks)
    assert "Alpha" in joined and "Bravo" in joined and "Charlie" in joined


def test_build_vector_store_embeds_all_chunks():
    """Every chunk in the store should have a non-empty embedding."""
    store = build_vector_store(FakeEmbedder())
    assert len(store) > 0
    assert all(len(c.embedding) > 0 for c in store.chunks)
    # chunks should carry their source filename
    assert all(c.source.endswith((".md", ".txt")) for c in store.chunks)
