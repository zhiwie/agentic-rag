"""
Tests for citation handling and the robust JSON parser.
"""

from src.agent import format_context, build_citations
from src.vectorstore import Chunk
from src.llm import parse_json


def sample_chunks():
    return [
        Chunk(text="Alpha fact.", source="a.md", chunk_id=0),
        Chunk(text="Bravo fact.", source="b.md", chunk_id=1),
    ]


def test_context_is_numbered_for_citation():
    """Context must label chunks [1], [2] so the model can cite them."""
    ctx = format_context(sample_chunks())
    assert "[1]" in ctx and "[2]" in ctx
    assert "a.md" in ctx and "b.md" in ctx


def test_build_citations_structure():
    """Citations expose number, source, and text for the UI."""
    cites = build_citations(sample_chunks())
    assert cites[0]["n"] == 1 and cites[0]["source"] == "a.md"
    assert cites[1]["n"] == 2 and cites[1]["text"] == "Bravo fact."


def test_parse_json_handles_markdown_fences():
    """The parser must survive LLMs that wrap JSON in ```json fences or text."""
    assert parse_json('```json\n{"route": "retrieve"}\n```')["route"] == "retrieve"
    assert parse_json('Sure! {"relevant": true} done')["relevant"] is True
    assert parse_json("not json at all") == {}   # degrades gracefully
