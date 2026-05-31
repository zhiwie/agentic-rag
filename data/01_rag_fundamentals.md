# Retrieval-Augmented Generation (RAG) Fundamentals

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines a large language
model (LLM) with an external knowledge source. Instead of relying only on the
information baked into the model's weights during training, a RAG system retrieves
relevant documents at query time and provides them to the LLM as context. This lets
the model answer questions using up-to-date or private information that it never saw
during training.

## Why RAG matters

LLMs have two well-known weaknesses: they can hallucinate (state false information
confidently), and their knowledge is frozen at their training cutoff date. RAG
addresses both problems. By grounding the model's answer in retrieved source
documents, RAG reduces hallucination and allows the knowledge base to be updated
without retraining the model. This makes RAG far cheaper than fine-tuning when the
goal is simply to give the model access to new facts.

## The core RAG pipeline

A traditional RAG pipeline has the following stages:

1. Ingestion: documents are loaded and split into smaller pieces called chunks.
2. Embedding: each chunk is converted into a numerical vector using an embedding model.
3. Indexing: the vectors are stored in a vector database for fast similarity search.
4. Retrieval: at query time, the user's question is embedded and the most similar
   chunks are retrieved.
5. Generation: the retrieved chunks are inserted into the prompt and the LLM generates
   an answer grounded in that context.

## Chunking

Chunking is the process of splitting long documents into smaller passages. Good
chunking is critical: chunks that are too large dilute the relevant information and
waste context window space, while chunks that are too small lose surrounding meaning.
A common strategy is to split on paragraph or sentence boundaries with a fixed target
size (for example, 500 to 1000 characters) and a small overlap between consecutive
chunks so that ideas spanning a boundary are not lost.

## Limitations of traditional RAG

Traditional (also called "naive") RAG retrieves once and generates once. It has no way
to recover if the first retrieval returns irrelevant chunks. It cannot decide whether
retrieval is even necessary for a given question, it cannot reformulate a poorly worded
query, and it cannot verify whether its own answer is actually supported by the
retrieved evidence. These limitations are what agentic RAG is designed to solve.
