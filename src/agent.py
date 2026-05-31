"""
agent.py
========
The agentic RAG controller. This is the part that makes the system "agentic"
rather than a one-shot pipeline.

Think of it as a graph of decision NODES connected by EDGES:

        +----------------+
        |  analyze_query |   route?
        +----------------+
           |          \
       retrieve      direct_answer
           |
       +----------+
       | retrieve |<---------------------+
       +----------+                      |
           |                             | (rewrite & loop,
       +----------------+                |  up to MAX loops)
       | grade_documents|                |
       +----------------+                |
           |   enough relevant chunks?   |
        no | -----------> transform_query+
       yes |
       +----------+
       | generate |
       +----------+
           |
       +----------------+
       | check_grounded |  (bonus: hallucination guard)
       +----------------+
           |
         ANSWER

Each node is a small function that reads and updates a shared `state` dict.
The driver function `run` wires them together and records a `trace` of every
step so the UI can show what the agent actually did.
"""

from __future__ import annotations
from typing import List

from . import config, prompts
from .llm import parse_json


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def analyze_query(state: dict, chat) -> dict:
    """Decide whether to retrieve, and produce a clean search query."""
    out = parse_json(chat.complete(prompts.ROUTER_PROMPT.format(question=state["question"])))
    state["route"] = out.get("route", "retrieve")
    state["search_query"] = out.get("search_query") or state["question"]
    state["trace"].append(
        f"Router decided: route='{state['route']}', search query='{state['search_query']}'"
    )
    return state


def retrieve(state: dict, store, embedder) -> dict:
    """Embed the search query and fetch the top-k chunks from the store."""
    query_vec = embedder.embed([state["search_query"]])[0]
    results = store.search(query_vec, top_k=config.TOP_K)
    state["candidates"] = [chunk for chunk, score in results]
    scores = ", ".join(f"{s:.2f}" for _, s in results)
    state["trace"].append(
        f"Retrieved {len(state['candidates'])} chunks (similarity scores: {scores})"
    )
    return state


def grade_documents(state: dict, chat) -> dict:
    """Keep only the chunks the LLM judges relevant to the question."""
    relevant = []
    for chunk in state["candidates"]:
        out = parse_json(
            chat.complete(prompts.GRADER_PROMPT.format(question=state["question"], chunk=chunk.text))
        )
        if out.get("relevant") is True:
            relevant.append(chunk)
    state["documents"] = relevant
    state["trace"].append(
        f"Graded chunks: {len(relevant)} of {len(state['candidates'])} judged relevant"
    )
    return state


def transform_query(state: dict, chat) -> dict:
    """Rewrite the search query to try to retrieve better chunks next time."""
    out = parse_json(
        chat.complete(
            prompts.REWRITE_PROMPT.format(
                question=state["question"], search_query=state["search_query"]
            )
        )
    )
    new_query = out.get("search_query") or state["search_query"]
    state["trace"].append(f"Rewrote search query: '{state['search_query']}' -> '{new_query}'")
    state["search_query"] = new_query
    return state


def generate(state: dict, chat) -> dict:
    """Generate an answer grounded in the relevant chunks, with [n] citations."""
    context = format_context(state["documents"])
    state["answer"] = chat.complete(
        prompts.GENERATE_PROMPT.format(context=context, question=state["question"])
    )
    state["trace"].append("Generated answer from relevant chunks")
    return state


def check_grounded(state: dict, chat) -> bool:
    """Bonus: verify the answer is supported by the sources. Returns True/False."""
    if not state["documents"]:
        return True  # nothing to check against; handled elsewhere
    context = format_context(state["documents"])
    out = parse_json(
        chat.complete(prompts.GROUNDED_PROMPT.format(context=context, answer=state["answer"]))
    )
    grounded = out.get("grounded", True)
    state["trace"].append(f"Groundedness check: {'passed' if grounded else 'FAILED'}")
    return grounded


def direct_answer(state: dict, chat) -> dict:
    """Answer without retrieval (greetings, off-topic messages)."""
    state["answer"] = chat.complete(prompts.DIRECT_PROMPT.format(question=state["question"]))
    state["documents"] = []
    state["trace"].append("Answered directly without retrieval")
    return state


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def format_context(documents: List) -> str:
    """Number the chunks [1], [2], ... so the model can cite them."""
    lines = []
    for i, chunk in enumerate(documents, start=1):
        lines.append(f"[{i}] (from {chunk.source})\n{chunk.text}")
    return "\n\n".join(lines)


def build_citations(documents: List) -> List[dict]:
    """A clean list of sources for the UI to display under the answer."""
    return [
        {"n": i, "source": chunk.source, "text": chunk.text}
        for i, chunk in enumerate(documents, start=1)
    ]


# ---------------------------------------------------------------------------
# DRIVER: wires the nodes together into the agent loop
# ---------------------------------------------------------------------------
def run(question: str, store, chat, embedder) -> dict:
    """
    Execute the full agentic RAG loop for one question and return the final
    state, including the answer, the citations, and a human-readable trace.
    """
    state = {
        "question": question,
        "search_query": question,
        "route": "retrieve",
        "candidates": [],
        "documents": [],
        "answer": "",
        "trace": [],
    }

    # Step 1: routing.
    analyze_query(state, chat)
    if state["route"] == "direct":
        direct_answer(state, chat)
        state["citations"] = []
        return state

    # Steps 2-4: retrieve -> grade -> (rewrite & loop if needed).
    for loop in range(config.MAX_RETRIEVAL_LOOPS + 1):
        retrieve(state, store, embedder)
        grade_documents(state, chat)
        if state["documents"]:
            break  # we have relevant chunks; stop looping.
        if loop < config.MAX_RETRIEVAL_LOOPS:
            transform_query(state, chat)  # correct course and try again.
        else:
            state["trace"].append("No relevant chunks found after retries")

    # Step 5: handle the case where nothing relevant was ever found.
    if not state["documents"]:
        state["answer"] = (
            "I could not find information about that in the knowledge base. "
            "Try rephrasing, or ask about RAG, vector databases, storage, "
            "fine-tuning, or agentic AI."
        )
        state["citations"] = []
        return state

    # Step 6: generate, then verify groundedness (one corrective retry).
    generate(state, chat)
    if not check_grounded(state, chat):
        state["trace"].append("Regenerating answer after groundedness failure")
        generate(state, chat)

    state["citations"] = build_citations(state["documents"])
    return state
