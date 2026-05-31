"""
ingest.py
=========
Turns raw documents into a searchable vector store. This is the offline part of
RAG that runs once, before any questions are asked.

Pipeline: load files -> split into chunks -> embed chunks -> store vectors.
"""

from __future__ import annotations
import os
from typing import List

from . import config
from .vectorstore import Chunk, VectorStore


def _read_pdf(path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    import fitz  # pymupdf
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n\n".join(text)


def _read_docx(path: str) -> str:
    """Extract all paragraph text from a Word document."""
    from docx import Document
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_documents(data_dir: str) -> List[tuple]:
    """Read .md, .txt, .pdf, and .docx files. Returns (filename, text) pairs."""
    readers = {
        ".md":   lambda p: open(p, "r", encoding="utf-8").read(),
        ".txt":  lambda p: open(p, "r", encoding="utf-8").read(),
        ".pdf":  _read_pdf,
        ".docx": _read_docx,
    }
    docs = []
    for name in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in readers:
            path = os.path.join(data_dir, name)
            try:
                text = readers[ext](path)
                if text.strip():
                    docs.append((name, text))
                    print(f"  Loaded: {name} ({len(text)} chars)")
            except Exception as e:
                print(f"  Warning: could not read {name}: {e}")
    return docs


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks of roughly `size` characters, breaking on
    paragraph boundaries where possible so chunks stay readable. Overlap keeps
    context that spans a boundary from being lost.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph keeps us under the size limit, append it.
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Start a new chunk, carrying over the tail of the previous one.
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                current = (tail + "\n\n" + para).strip()
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def build_vector_store(embedder, data_dir: str = None) -> VectorStore:
    """Run the full ingestion pipeline and return a ready-to-search store."""
    data_dir = data_dir or config.DATA_DIR
    store = VectorStore()

    all_chunks: List[Chunk] = []
    for filename, text in load_documents(data_dir):
        pieces = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            all_chunks.append(Chunk(text=piece, source=filename, chunk_id=i))

    # Embed all chunk texts in one batch, then attach the vectors.
    embeddings = embedder.embed([c.text for c in all_chunks])
    for chunk, emb in zip(all_chunks, embeddings):
        chunk.embedding = emb

    store.add(all_chunks)
    return store
