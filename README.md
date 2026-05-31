# Question 1: Agentic RAG

An Agentic RAG system retrieves information from a knowledge base before generating its response.

When a question is asked, it searches the knowledge base for relevant document chunks. The retrieved chunks are reviewed, and only useful ones are kept. If the results are poor, the system rewrites the search query and tries again. Once relevant information is found, it generates an answer and includes citations to the source documents. Before returning the response, it performs a final validation step to ensure the answer is supported by the retrieved content.

---

## Project Structure

```
agentic-rag/
├── app.py                 ← the Streamlit web app (the live demo)
├── data/                  ← 5 documents the system answers questions about
├── src/
│   ├── config.py          ← all settings in one place
│   ├── prompts.py         ← every prompt the agent uses
│   ├── llm.py             ← OpenAI wrappers + fakes for testing
│   ├── vectorstore.py     ← ChromaDB-backed store + similarity search
│   ├── ingest.py          ← loads docs, chunks them, builds the store
│   └── agent.py           ← THE CORE: the agent loop
├── eval/                  ← retrieval evaluation (accuracy + latency)
│   ├── eval_set.json      ← 10 labeled questions
│   └── run_eval.py        ← reports Hit@k, MRR, latency
├── tests/                 ← 15 automated tests (no API key needed)
├── requirements.txt
└── .env
```

---

## How to Run it

### 1. Install Python 3.10 or newer
Check by opening a terminal and typing:
```bash
python --version
```
If you see a version number (e.g. `Python 3.11`), you're good.

### 2. Open a terminal in this folder
Navigate into the project:
```bash
cd path/to/agentic-rag
```

### 3. Create a virtual environment
This keeps the project's packages separate from the rest of your computer.
```bash
python -m venv venv
source venv/Scripts/activate
```
You'll know it worked when your terminal line starts with `(venv)`.

### 4. Install the required packages
```bash
pip install -r requirements.txt
```

### 5. Add your OpenAI API key
Get one at https://platform.openai.com/api-keys, then set it:
```bash
export OPENAI_API_KEY="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"     

```
The app will still **open** without an API Key; For real answers, you will need to export the real API Key.

### 6. Run the tests (proves everything works)
```bash
python -m pytest tests/ -v
```
You should see `15 passed`.

### 7. Launch the app
```bash
streamlit run app.py
```
Your browser opens automatically. Type a question and click **Ask**.

### 8. (Optional) Run the retrieval evaluation
```bash
python -m eval.run_eval
```