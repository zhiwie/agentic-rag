"""
Tests for the agent's control flow. We use FakeChat, a scripted LLM, to force
specific decisions and verify the agent routes, grades, loops, and self-corrects
correctly. This is where we prove the system is genuinely "agentic".
"""

import json
from src.ingest import build_vector_store
from src.llm import FakeChat, FakeEmbedder
from src import agent


def make_store():
    return build_vector_store(FakeEmbedder())


def test_direct_route_skips_retrieval():
    """A greeting should be routed 'direct' and never retrieve."""
    chat = FakeChat(responses=[
        json.dumps({"route": "direct", "search_query": ""}),  # router
        "Hello! Ask me about RAG, storage, or agentic AI.",   # direct answer
    ])
    result = agent.run("hello there", make_store(), chat, FakeEmbedder())
    assert result["citations"] == []
    assert "Answered directly without retrieval" in " ".join(result["trace"])


def test_happy_path_generates_citations():
    """When chunks grade relevant, the agent generates and attaches citations."""
    # router -> retrieve; then grade TOP_K chunks all relevant; generate; grounded.
    responses = [json.dumps({"route": "retrieve", "search_query": "ssd controller"})]
    responses += [json.dumps({"relevant": True})] * 4   # grade 4 candidates
    responses += ["An SSD controller manages the flash [1]."]  # generate
    responses += [json.dumps({"grounded": True})]       # groundedness check
    chat = FakeChat(responses=responses)
    result = agent.run("what does a controller do", make_store(), chat, FakeEmbedder())
    assert "[1]" in result["answer"]
    assert len(result["citations"]) >= 1


def test_self_correction_loop_runs_on_irrelevant_chunks():
    """
    If the first grading round finds nothing relevant, the agent must rewrite the
    query and retry (the corrective loop), then succeed on the second round.
    """
    responses = [json.dumps({"route": "retrieve", "search_query": "vague query"})]
    responses += [json.dumps({"relevant": False})] * 4          # round 1: all irrelevant
    responses += [json.dumps({"search_query": "better query"})] # rewrite
    responses += [json.dumps({"relevant": True})] * 4           # round 2: relevant
    responses += ["Answer grounded in sources [1]."]            # generate
    responses += [json.dumps({"grounded": True})]               # grounded check
    chat = FakeChat(responses=responses)
    result = agent.run("ambiguous question", make_store(), chat, FakeEmbedder())
    trace = " ".join(result["trace"])
    assert "Rewrote search query" in trace        # proves it self-corrected
    assert len(result["citations"]) >= 1


def test_gives_up_gracefully_when_nothing_relevant():
    """After exhausting retries with no relevant chunks, answer gracefully."""
    responses = [json.dumps({"route": "retrieve", "search_query": "q"})]
    # MAX_RETRIEVAL_LOOPS=2 -> 3 grading rounds, each all-irrelevant, 2 rewrites
    responses += [json.dumps({"relevant": False})] * 4
    responses += [json.dumps({"search_query": "q2"})]
    responses += [json.dumps({"relevant": False})] * 4
    responses += [json.dumps({"search_query": "q3"})]
    responses += [json.dumps({"relevant": False})] * 4
    chat = FakeChat(responses=responses)
    result = agent.run("unanswerable", make_store(), chat, FakeEmbedder())
    assert result["citations"] == []
    assert "could not find" in result["answer"].lower()


def test_groundedness_failure_triggers_regeneration():
    """If the groundedness check fails once, the agent regenerates the answer."""
    responses = [json.dumps({"route": "retrieve", "search_query": "ssd"})]
    responses += [json.dumps({"relevant": True})] * 4
    responses += ["First (hallucinated) answer [1]."]   # first generate
    responses += [json.dumps({"grounded": False})]      # fails check
    responses += ["Second (corrected) answer [1]."]     # regenerate
    chat = FakeChat(responses=responses)
    result = agent.run("question", make_store(), chat, FakeEmbedder())
    assert "Second" in result["answer"]
    assert "Regenerating answer after groundedness failure" in " ".join(result["trace"])
