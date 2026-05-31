# Vector Databases and Embeddings

## What is an embedding?

An embedding is a list of numbers (a vector) that represents the meaning of a piece of
text. An embedding model is trained so that texts with similar meaning are mapped to
vectors that are close together in space, while unrelated texts are far apart. A
typical embedding vector has hundreds or thousands of dimensions. For example, OpenAI's
text-embedding-3-small model produces vectors with 1536 dimensions.

## Measuring similarity

Once text is represented as vectors, we can measure how similar two pieces of text are
by measuring the distance or angle between their vectors. The most common measure is
cosine similarity, which computes the cosine of the angle between two vectors. A cosine
similarity of 1.0 means the vectors point in exactly the same direction (very similar
meaning), 0.0 means they are unrelated, and -1.0 means they are opposite. Because it
looks at direction rather than magnitude, cosine similarity works well for comparing
text embeddings.

## What is a vector database?

A vector database stores embedding vectors and supports fast similarity search. Given a
query vector, it returns the stored vectors that are closest to it. For small datasets,
a brute-force search that compares the query against every stored vector is fast enough.
For large datasets with millions of vectors, specialized index structures such as HNSW
(Hierarchical Navigable Small World) graphs are used to find approximate nearest
neighbors quickly without checking every vector.

## Popular vector stores

Common vector databases and libraries include Chroma, FAISS, Pinecone, Weaviate, and
pgvector (an extension for PostgreSQL). They differ in whether they run in-process or as
a separate server, whether they persist data to disk, and how they scale. For a small
prototype, an in-memory store using brute-force cosine similarity is simple, transparent,
and perfectly adequate. Production systems with large corpora typically move to a
dedicated vector database with approximate nearest neighbor indexing.

## Top-k retrieval

Retrieval usually returns the top-k most similar chunks, where k is a small number such
as 3 to 5. Choosing k involves a trade-off: a larger k increases the chance that the
relevant chunk is included, but it also adds noise and consumes more of the LLM's
context window. Many systems retrieve a slightly larger k and then re-rank or filter the
results to keep only the most relevant chunks.
