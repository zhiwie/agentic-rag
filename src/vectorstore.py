"""
vectorstore.py
==============
A vector store backed by ChromaDB, a popular open-source vector database.

Design notes:
- We compute embeddings ourselves (via the pluggable embedder in llm.py) and
  hand the precomputed vectors to Chroma. This keeps the embedder swappable and
  lets the test suite use a deterministic fake embedder with no API key.
- We use Chroma's cosine space, so "closeness" matches the cosine similarity we
  discuss in the presentation. Chroma returns a cosine *distance*; we convert it
  back to a similarity score (similarity = 1 - distance) for display.
- We use an in-memory (ephemeral) client so every run starts clean — ideal for a
  demo. Switching to a persistent on-disk store is a one-line change
  (PersistentClient(path=...)), noted below.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import math
import uuid

import chromadb


@dataclass
class Chunk:
    """One retrievable passage of text plus where it came from."""
    text: str
    source: str          # filename the chunk came from
    chunk_id: int        # position of the chunk within that file
    embedding: List[float] = field(default_factory=list)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine of the angle between two vectors: 1.0 = identical direction.

    Kept as a standalone helper because it is used by the evaluation harness and
    the unit tests, and because it documents exactly what Chroma is doing for us
    under the hood in cosine space.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """A thin wrapper around a Chroma collection."""

    def __init__(self):
        # EphemeralClient keeps everything in memory. For persistence across
        # runs, swap this line for:  chromadb.PersistentClient(path="./chroma")
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(
            name=f"kb_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},   # use cosine distance
        )
        self._chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        """Store chunks and their precomputed embeddings in Chroma."""
        if not chunks:
            return
        self._collection.add(
            ids=[str(len(self._chunks) + i) for i in range(len(chunks))],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_id": c.chunk_id} for c in chunks],
        )
        self._chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 4):
        """Return the top_k most similar chunks as (Chunk, similarity) pairs."""
        n = min(top_k, len(self._chunks)) or 1
        res = self._collection.query(query_embeddings=[query_embedding], n_results=n)
        out = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for text, meta, dist in zip(docs, metas, dists):
            chunk = Chunk(text=text, source=meta["source"], chunk_id=meta["chunk_id"])
            similarity = 1.0 - dist          # cosine distance -> similarity
            out.append((chunk, similarity))
        return out

    @property
    def chunks(self) -> List[Chunk]:
        """All stored chunks (read-only view), e.g. for inspection or tests."""
        return self._chunks

    def __len__(self) -> int:
        return len(self._chunks)
