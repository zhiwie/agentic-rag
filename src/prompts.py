"""
prompts.py
==========
Every prompt the agent uses, kept in one file so they are easy to read,
tweak, and explain. Each prompt corresponds to one decision the agent makes.
"""

# 1. ROUTING / QUERY ANALYSIS
# The agent first decides whether a question needs the knowledge base at all,
# and produces a clean search query. This is something traditional RAG cannot do.
ROUTER_PROMPT = """You are the routing component of a retrieval system.
The knowledge base covers these topics:
- Retrieval-Augmented Generation (RAG)
- Vector databases and embeddings
- NAND flash and SSD storage
- LLM fine-tuning
- Agentic AI systems

Given the user's question, decide:
1. "route": either "retrieve" (the question is about the topics above and needs
   the knowledge base) or "direct" (it is a greeting, chit-chat, or clearly
   unrelated, and should be answered without retrieval).
2. "search_query": a concise, keyword-rich version of the question, optimised
   for similarity search. If route is "direct", set this to an empty string.

Respond with ONLY a JSON object, no other text, in this exact form:
{{"route": "retrieve" or "direct", "search_query": "..."}}

User question: {question}"""


# 2. DOCUMENT GRADING
# After retrieval, the agent grades whether each chunk is actually relevant.
# This filters out noise and is the basis of the self-correction loop.
GRADER_PROMPT = """You are grading whether a retrieved document chunk is relevant
to a user's question. Be strict: a chunk is relevant only if it contains
information that helps answer the question.

User question: {question}

Document chunk:
{chunk}

Respond with ONLY a JSON object in this exact form:
{{"relevant": true or false}}"""


# 3. QUERY REWRITING
# If retrieval produced too few relevant chunks, the agent rewrites the query
# and tries again. This is the agent "correcting course".
REWRITE_PROMPT = """The previous search did not return enough relevant results.
Rewrite the search query to be clearer and use different keywords or synonyms
that might match the source documents better.

Original question: {question}
Previous search query: {search_query}

Respond with ONLY a JSON object in this exact form:
{{"search_query": "..."}}"""


# 4. ANSWER GENERATION WITH CITATIONS
# The agent answers using ONLY the retrieved chunks and cites them by number.
GENERATE_PROMPT = """You are a helpful assistant answering a question using ONLY
the numbered sources provided. Follow these rules strictly:
- Use only information found in the sources. Do not add outside knowledge.
- After each sentence or claim, cite the source(s) it came from using square
  brackets, like [1] or [2][3].
- If the sources do not contain enough information to answer, say so plainly
  instead of guessing.
- Keep the answer clear and concise.

Sources:
{context}

Question: {question}

Answer (with [n] citations):"""


# 5. GROUNDEDNESS CHECK (bonus: hallucination guard)
# The agent verifies its own answer is supported by the sources before
# returning it. This is reflection / self-verification.
GROUNDED_PROMPT = """You are checking whether an answer is fully supported by the
provided sources. An answer is grounded only if every factual claim in it can be
traced to the sources.

Sources:
{context}

Answer:
{answer}

Respond with ONLY a JSON object in this exact form:
{{"grounded": true or false}}"""


# 6. DIRECT ANSWER (no retrieval)
# Used when the router decides the question does not need the knowledge base.
DIRECT_PROMPT = """You are a friendly assistant for a document question-answering
system whose knowledge base covers RAG, vector databases, storage, LLM
fine-tuning, and agentic AI. The user's message does not require searching the
knowledge base. Respond briefly and, if appropriate, invite them to ask a
question about those topics.

User message: {question}"""
