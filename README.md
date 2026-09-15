# Stripe Policy RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) chatbot designed to answer user inquiries about Stripe's privacy, data governance, and legal policies with factual grounding and verifiable source citations.

---

## Architecture Overview

```
                        [User Prompt + Chat History]
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      Node 1: refine_query       │
                    │   (LLM query rewrite & expand)  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │        Node 2: retrieve         │
                    │    (FAISS MMR Search: k=5)      │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      Node 3: build_context      │
                    │  (Deduplicate & Token Truncate) │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │     Node 4: generate_answer     │
                    │    (Strictly Grounded LLM)      │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                        [Streamlit Chat UI + Sources]
```

### Ingestion Pipeline (Offline / On-Demand)
1. **Sources**: Target policy URLs (default: `https://stripe.com/in/privacy` and `https://stripe.com/in/legal/privacy-center`).
2. **Document Loader**: `WebBaseLoader` with `bs4.SoupStrainer` targeting main content containers (stripping headers, footers, navigation, scripts, and styles).
3. **Chunking**: `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=300`).
4. **Embeddings**: OpenAI `text-embedding-3-small`.
5. **Vector Store**: `FAISS` persisted locally in `./data/faiss_index/`.

### Query Pipeline (LangGraph StateGraph)
Orchestrated as an explicit state machine passing a typed `GraphState`:
- **`refine_query`**: Resolves conversational pronouns and reformulates user prompts into standalone semantic queries.
- **`retrieve`**: Uses Maximal Marginal Relevance (MMR) search (`k=5`, `fetch_k=20`, `lambda_mult=0.5`) to eliminate redundancy.
- **`build_context`**: Merges retrieved chunks with inline citations within a safe token budget.
- **`generate_answer`**: Enforces strict grounding with `gpt-4o-mini`, citing source URLs and refusing speculation outside retrieved context.

---

## Tech Stack
- **Python 3.11+**
- **LangGraph**: Stateful query pipeline orchestration
- **LangChain / LangChain-Community / LangChain-OpenAI**: Loaders, splitters, FAISS, and model interfaces
- **FAISS (`faiss-cpu`)**: Vector database with MMR similarity search
- **Pydantic (v2) & Pydantic-Settings**: Schema validation and environment-based configuration
- **Streamlit**: Interactive web chat interface
- **Pytest**: Automated test suite

---

## Project Structure

```
stripe-privacy-rag-chatbot/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── app.py                      # Streamlit frontend entrypoint
├── config.py                   # Pydantic BaseSettings
├── data/
│   └── faiss_index/            # Persisted FAISS vector store
├── ingestion/
│   ├── __init__.py
│   ├── loader.py               # WebBaseLoader + SoupStrainer extraction
│   ├── splitter.py             # RecursiveCharacterTextSplitter + chunk IDs
│   └── build_index.py          # CLI index builder: load -> split -> embed -> save
├── rag/
│   ├── __init__.py
│   ├── state.py                # GraphState TypedDict
│   ├── prompts.py              # System prompts (Refinement + Grounded answer)
│   ├── vectorstore.py          # FAISS loader & MMR retriever factory
│   ├── nodes.py                # LangGraph nodes (refine, retrieve, context, generate)
│   └── graph.py                # StateGraph assembly & pipeline execution
├── models/
│   ├── __init__.py
│   └── schemas.py              # Pydantic schemas (ChunkMetadata, RetrievedChunk, etc.)
└── tests/
    ├── __init__.py
    ├── test_ingestion.py       # Unit tests for loader and splitter
    ├── test_vectorstore.py     # Unit tests for vectorstore and MMR
    └── test_graph.py           # Unit tests for end-to-end LangGraph flow
```

---

## Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/TousifTamboli/stripe-privacy-rag-chatbot.git
cd stripe-privacy-rag-chatbot

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

### 3. Build the FAISS Vector Index

Run the ingestion CLI script to scrape, chunk, embed, and persist the index:

```bash
python -m ingestion.build_index
```

### 4. Run the Streamlit Chatbot

Launch the interactive Streamlit chat interface:

```bash
streamlit run app.py
```

---

## Running Automated Tests

Run the test suite with pytest:

```bash
pytest -v
```

---

## Customization & Extension Points

### Adding Additional Policy URLs
You can expand the indexed policy documents by updating `SOURCE_URLS` in `config.py` (or through environment variables), and triggering a re-index either via:
```bash
python -m ingestion.build_index
```
or by clicking the **🔄 Rebuild Index** button directly in the Streamlit sidebar.

### Future Improvements & Architectural Seams
- **Hybrid Search**: Combine BM25 keyword search with FAISS dense vector search using LangChain's `EnsembleRetriever`.
- **Cross-Encoder Re-Ranking**: Plug in a re-ranker (e.g. Cohere Re-rank or FlashRank) between Node 2 (`retrieve`) and Node 3 (`build_context`) to re-order candidate passages.
- **Semantic Chunking**: Swap `RecursiveCharacterTextSplitter` with `SemanticChunker` (reference template provided in [`ingestion/splitter.py`](file:///Users/tousif/githubs/stripe-privacy-rag-chatbot/ingestion/splitter.py)).
