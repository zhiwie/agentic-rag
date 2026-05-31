"""
llm.py
======
Thin wrappers around the OpenAI API for (a) chat completions and (b) embeddings.

Each wrapper has a matching FAKE version used by the test suite. The fakes let
us test all the agent's logic (routing, grading, looping, citations) WITHOUT
needing an API key or spending money. This is good engineering practice:
we test our own logic, not OpenAI's servers.
"""

from __future__ import annotations
import hashlib
import json
import math
from typing import List

from . import config


# ---------------------------------------------------------------------------
# Real implementations (used when an OPENAI_API_KEY is set)
# ---------------------------------------------------------------------------
class OpenAIChat:
    """Wraps OpenAI chat completions."""

    def __init__(self, model: str = config.LLM_MODEL):
        from openai import OpenAI  # imported lazily so tests don't need it
        self.client = OpenAI()
        self.model = model

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


class OpenAIEmbedder:
    """Wraps OpenAI embeddings."""

    def __init__(self, model: str = config.EMBEDDING_MODEL):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# Fake implementations (used by tests, and as an offline fallback)
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """
    A deterministic embedder that needs no API. It turns text into a fixed-size
    vector using a hashing trick: each word is hashed into one of N buckets and
    the bucket count is incremented. Texts sharing words get similar vectors,
    so cosine similarity still behaves sensibly for testing retrieval.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FakeChat:
    """
    A scriptable fake LLM. You hand it a list of responses and it returns them
    in order on each call. Tests use this to simulate the LLM making specific
    decisions (e.g. "this chunk is relevant") so we can verify the agent's
    control flow deterministically.
    """

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.calls: List[str] = []

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        if not self._responses:
            return "{}"
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_json(text: str) -> dict:
    """
    LLMs sometimes wrap JSON in markdown fences or add stray text. This pulls
    out the first {...} block and parses it, returning {} on failure so the
    agent can degrade gracefully instead of crashing.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
