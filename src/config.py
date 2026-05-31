"""
config.py
=========
All the tunable settings for the Agentic RAG system live in one place.
Keeping configuration separate from logic makes the system easy to explain
and easy to adjust without digging through code.
"""

import os

# --- Model settings ---------------------------------------------------------
# The chat model used for reasoning, grading, and answer generation.
LLM_MODEL = "gpt-4o-mini"

# The embedding model used to turn text into vectors.
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Chunking settings ------------------------------------------------------
# Target size of each text chunk, in characters, and how much neighbouring
# chunks overlap. Overlap prevents ideas that span a boundary from being lost.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retrieval settings -----------------------------------------------------
# How many chunks to retrieve per search.
TOP_K = 4

# --- Agent settings ---------------------------------------------------------
# Maximum number of retrieve -> grade -> rewrite loops before the agent gives
# up and answers with whatever it has. This bounds cost and prevents infinite
# loops, which is essential for any agentic system.
MAX_RETRIEVAL_LOOPS = 2

# Where the knowledge-base documents live.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
