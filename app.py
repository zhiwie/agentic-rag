"""
app.py
======
The Streamlit web app used for the live demo. Run it with:

    streamlit run app.py

It builds the vector store once (cached), then for each question it runs the
agentic RAG loop and shows: the answer with citations, the cited sources, and
the agent's step-by-step trace so the audience can SEE the agentic behaviour.
"""

import os
from dotenv import load_dotenv
import streamlit as st

# Load .env file so OPENAI_API_KEY is set automatically on every run.
load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]

from src import config
from src.ingest import build_vector_store
from src.llm import OpenAIChat, OpenAIEmbedder, FakeChat, FakeEmbedder
from src import agent

st.set_page_config(page_title="Agentic RAG", page_icon="🔎", layout="wide")


# --- Build the system once and cache it ------------------------------------
@st.cache_resource
def load_system():
    """Create the LLM, embedder, and vector store. Cached across reruns."""
    if os.environ.get("OPENAI_API_KEY"):
        chat = OpenAIChat()
        embedder = OpenAIEmbedder()
        mode = "OpenAI (live)"
    else:
        # Offline fallback so the UI still loads without a key.
        chat = FakeChat(responses=[])
        embedder = FakeEmbedder()
        mode = "OFFLINE fallback (set OPENAI_API_KEY for real answers)"
    store = build_vector_store(embedder)
    return chat, embedder, store, mode


chat, embedder, store, mode = load_system()

# --- Header -----------------------------------------------------------------
st.title("🔎 Agentic RAG")
st.caption(
    f"Self-correcting retrieval over {len(store)} chunks  ·  Mode: {mode}"
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Route** – decide if retrieval is needed\n"
        "2. **Retrieve** – fetch top-k similar chunks\n"
        "3. **Grade** – keep only relevant chunks\n"
        "4. **Rewrite & retry** – if not enough relevant chunks\n"
        "5. **Generate** – answer with [n] citations\n"
        "6. **Check grounded** – verify against sources"
    )
    st.divider()
    st.markdown("**Knowledge base topics**")
    st.markdown(
        "- RAG fundamentals\n- Vector databases\n- NAND / SSD storage\n"
        "- LLM fine-tuning\n- Agentic AI"
    )
    st.divider()
    st.markdown("**Try asking**")
    st.markdown(
        "- What is the difference between traditional RAG and agentic RAG?\n"
        "- How does LoRA reduce fine-tuning cost?\n"
        "- What does an SSD controller do?\n"
        "- Hello!  *(watch it skip retrieval)*"
    )

# --- Chat input -------------------------------------------------------------
question = st.text_input("Ask a question:", placeholder="e.g. What is cosine similarity?")

if st.button("Ask", type="primary") and question:
    if mode.startswith("OFFLINE"):
        st.warning(
            "No OPENAI_API_KEY set, so the agent's LLM steps are disabled. "
            "Set your key (see README) to get real answers. Retrieval still works."
        )
    with st.spinner("Agent is thinking..."):
        result = agent.run(question, store, chat, embedder)

    st.subheader("Answer")
    st.write(result["answer"])

    if result.get("citations"):
        st.subheader("Sources")
        for c in result["citations"]:
            with st.expander(f"[{c['n']}] {c['source']}"):
                st.write(c["text"])

    st.subheader("🧠 Agent trace")
    for i, step in enumerate(result["trace"], start=1):
        st.markdown(f"`{i}.` {step}")
