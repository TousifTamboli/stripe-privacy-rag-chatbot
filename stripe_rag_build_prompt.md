# Build Prompt: Stripe Policy RAG Chatbot

Copy everything below into Claude Code (or use it as your own spec) to build the project end-to-end.

---

## PROJECT BRIEF

Build a Retrieval-Augmented Generation (RAG) chatbot that answers user questions about Stripe's
privacy and legal policies. The bot must ingest two web pages, chunk and embed them, store vectors
in FAISS, retrieve relevant context for a user query, refine the query with an LLM, and generate a
grounded answer with citations. Frontend is Streamlit. Backend is Python using LangGraph as the
orchestration layer (LangChain used only for utility components: loaders, splitters, embeddings
wrapper, FAISS wrapper). Pydantic is used for all structured data models and config validation.
OpenAI is the LLM + embedding provider. No deployment yet — local run only (`streamlit run app.py`).

### Data sources (hardcoded for now, but structured to allow more later)
- https://stripe.com/in/privacy
- https://stripe.com/in/legal/privacy-center

---

## ARCHITECTURE (must match this exactly)

**Ingestion pipeline (run once / on-demand via a script or Streamlit "Rebuild Index" button):**

1. `Sources` → list of URLs
2. `Document Loader` → `WebBaseLoader` (LangChain) for each URL. Use `bs4.SoupStrainer` to strip
   nav/footer/script/style tags and keep only main content divs — reduces noise before chunking.
3. `Text Splitter` → `RecursiveCharacterTextSplitter` (chunk_size=1000, chunk_overlap=300), as shown
   in the diagram ("Currently using this one"). Keep `SemanticChunker` as a commented-out alternative
   for later experimentation, but ship with RecursiveCharacterTextSplitter.
4. Output: list of `chunks`, each chunk retains metadata: `source_url`, `chunk_id`, `title` (if
   extractable).
5. `Embedding Model` → OpenAI `text-embedding-3-small` (cheap, good quality; make model name
   configurable via env var).
6. `Vector Store` → FAISS (`langchain_community.vectorstores.FAISS`), persisted to local disk
   (`./data/faiss_index/`) so it doesn't need to be rebuilt every run.

**Query pipeline (runs per user message, orchestrated as a LangGraph graph):**

1. `user_prompt` → raw user input from Streamlit chat box.
2. `LLM (query refiner node)` → rewrites/expands the user's raw question into a clearer, more
   retrieval-friendly `refined_prompt` (handles pronoun resolution using chat history, expands
   abbreviations, fixes ambiguity). This is a LangGraph node with its own small system prompt.
3. `retriever` → FAISS retriever built with **MMR** (Maximal Marginal Relevance) search as shown in
   the diagram — `search_type="mmr"`, `k=5`, `fetch_k=20`, `lambda_mult=0.5` — to reduce redundant
   chunks and diversify context. Retrieves top chunks using `refined_prompt` via semantic search.
4. `Context` → concatenated retrieved chunks (with source attribution) assembled into a single
   context block, deduplicated, and truncated to a max token budget.
5. `Context + refined Prompt` → combined into the final prompt via a Pydantic-validated prompt
   template (system message includes grounding instructions: answer only from context, cite the
   source URL, say "I don't know" if not in context).
6. `Prompt` → final assembled message list sent to LLM.
7. `LLM (answer generation node)` → OpenAI chat model (`gpt-4o-mini` default, configurable) generates
   the final answer.
8. `Response` → returned to Streamlit, rendered in chat, with an expandable "Sources" section showing
   which chunks/URLs were used.

This entire query pipeline (steps 2–8) is implemented as a **LangGraph StateGraph** with nodes:
`refine_query → retrieve → build_context → generate_answer → END`. State is a Pydantic-backed
TypedDict passed between nodes.

---

## TECH STACK

- Python 3.11+
- `langgraph` — orchestration of the query pipeline as a state graph
- `langchain`, `langchain-community`, `langchain-openai` — loaders, text splitter, FAISS wrapper,
  OpenAI embeddings/chat wrapper
- `faiss-cpu` — vector store
- `pydantic` (v2) — config models, graph state schema, structured LLM outputs
- `openai` — via langchain-openai (do not call raw SDK directly, keep one abstraction layer)
- `streamlit` — frontend chat UI
- `python-dotenv` — env var loading
- `beautifulsoup4` — used internally by WebBaseLoader + SoupStrainer

---

## PROJECT STRUCTURE

```
stripe-rag/
├── .env.example
├── requirements.txt
├── README.md
├── app.py                      # Streamlit entrypoint
├── config.py                   # Pydantic Settings (env-driven config)
├── data/
│   └── faiss_index/             # persisted FAISS index (gitignored)
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                # WebBaseLoader + SoupStrainer logic
│   ├── splitter.py               # RecursiveCharacterTextSplitter setup
│   └── build_index.py            # CLI script: load -> split -> embed -> save FAISS
├── rag/
│   ├── __init__.py
│   ├── state.py                  # Pydantic/TypedDict GraphState schema
│   ├── nodes.py                  # refine_query, retrieve, build_context, generate_answer
│   ├── graph.py                  # LangGraph StateGraph wiring
│   ├── prompts.py                # All system/user prompt templates as constants
│   └── vectorstore.py            # FAISS load/save/retriever factory (MMR configured here)
├── models/
│   ├── __init__.py
│   └── schemas.py                # Pydantic models: ChunkMetadata, RetrievedChunk, ChatMessage, etc.
└── tests/
    ├── test_ingestion.py
    ├── test_graph.py
    └── test_vectorstore.py
```

---

## DETAILED REQUIREMENTS BY FILE

### `config.py`
- Pydantic `BaseSettings` class `Settings` loading from `.env`:
  - `OPENAI_API_KEY: str`
  - `EMBEDDING_MODEL: str = "text-embedding-3-small"`
  - `CHAT_MODEL: str = "gpt-4o-mini"`
  - `CHUNK_SIZE: int = 1000`
  - `CHUNK_OVERLAP: int = 300`
  - `RETRIEVER_K: int = 5`
  - `RETRIEVER_FETCH_K: int = 20`
  - `MMR_LAMBDA: float = 0.5`
  - `FAISS_INDEX_PATH: str = "./data/faiss_index"`
  - `SOURCE_URLS: list[str]` default = the two Stripe URLs
- Singleton `settings = Settings()` instance imported everywhere else.

### `models/schemas.py`
- `ChunkMetadata(BaseModel)`: `source_url: str`, `chunk_id: str`, `title: str | None`
- `RetrievedChunk(BaseModel)`: `content: str`, `metadata: ChunkMetadata`, `score: float`
- `ChatMessage(BaseModel)`: `role: Literal["user","assistant"]`, `content: str`
- `RAGResponse(BaseModel)`: `answer: str`, `sources: list[str]`, `retrieved_chunks: list[RetrievedChunk]`

### `ingestion/loader.py`
- Function `load_documents(urls: list[str]) -> list[Document]`
- Use `WebBaseLoader` with `bs_kwargs={"parse_only": SoupStrainer(...)}` targeting main content
  containers (fallback to full page if selector fails). Set a proper `User-Agent` header.
- Attach `source_url` metadata to each doc.

### `ingestion/splitter.py`
- Function `split_documents(docs) -> list[Document]` using `RecursiveCharacterTextSplitter`
  with `chunk_size=settings.CHUNK_SIZE`, `chunk_overlap=settings.CHUNK_OVERLAP`,
  `separators=["\n\n", "\n", ". ", " ", ""]`.
- Assign a unique `chunk_id` (uuid or index-based) to each chunk's metadata.
- Include a commented-out `SemanticChunker` alternative block for future swap-in.

### `ingestion/build_index.py`
- CLI script (`python -m ingestion.build_index`) that:
  1. Loads docs from `settings.SOURCE_URLS`
  2. Splits into chunks
  3. Embeds with `OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)`
  4. Builds `FAISS.from_documents(chunks, embeddings)`
  5. Saves to `settings.FAISS_INDEX_PATH` via `.save_local()`
  6. Prints chunk count + sample chunk for sanity check
- Idempotent — safe to re-run to rebuild index from scratch.

### `rag/vectorstore.py`
- `load_vectorstore() -> FAISS` — loads persisted index (raise clear error if missing, telling user
  to run `build_index.py` first)
- `get_retriever(vectorstore) -> VectorStoreRetriever` — configured with:
  ```python
  vectorstore.as_retriever(
      search_type="mmr",
      search_kwargs={"k": settings.RETRIEVER_K, "fetch_k": settings.RETRIEVER_FETCH_K,
                     "lambda_mult": settings.MMR_LAMBDA}
  )
  ```

### `rag/state.py`
- `GraphState(TypedDict)`:
  - `user_prompt: str`
  - `chat_history: list[ChatMessage]`
  - `refined_prompt: str`
  - `retrieved_chunks: list[RetrievedChunk]`
  - `context: str`
  - `answer: str`
  - `sources: list[str]`

### `rag/prompts.py`
Define as constants:
- `QUERY_REFINEMENT_SYSTEM_PROMPT` — instructs the LLM to rewrite the user's question standalone
  (resolve pronouns using chat history), keep it concise, don't answer it, just rewrite for retrieval.
- `ANSWER_SYSTEM_PROMPT` — strict grounding instructions: "Answer ONLY using the provided context
  from Stripe's policy pages. If the answer isn't in the context, say you don't have that
  information. Always cite which source URL(s) support your answer. Be concise and accurate — this
  concerns legal/financial policy, do not speculate."

### `rag/nodes.py`
Four LangGraph node functions, each taking and returning `GraphState`:
1. `refine_query(state)` — calls chat model with `QUERY_REFINEMENT_SYSTEM_PROMPT` + chat history +
   `user_prompt` → sets `refined_prompt`
2. `retrieve(state)` — calls retriever with `refined_prompt` → sets `retrieved_chunks` (map LangChain
   `Document` results into `RetrievedChunk` pydantic objects, dedupe by chunk_id)
3. `build_context(state)` — concatenates `retrieved_chunks` content with inline source markers like
   `[Source: <url>]`, truncates to a max char/token budget → sets `context` and `sources`
4. `generate_answer(state)` — calls chat model with `ANSWER_SYSTEM_PROMPT` + `context` +
   `refined_prompt` → sets `answer`

### `rag/graph.py`
- Build `StateGraph(GraphState)`
- Add nodes: `refine_query`, `retrieve`, `build_context`, `generate_answer`
- Edges: `START → refine_query → retrieve → build_context → generate_answer → END`
- Compile and export `rag_graph = graph.compile()`
- Expose a single function `run_rag_pipeline(user_prompt: str, chat_history: list[ChatMessage]) -> RAGResponse`
  that invokes the compiled graph and maps the final state into a `RAGResponse`.

### `app.py` (Streamlit)
- Page config: title "Stripe Policy Assistant"
- Sidebar:
  - Show which two source URLs are indexed
  - "Rebuild Index" button → runs `ingestion.build_index` logic with a spinner
  - Show index status (exists / chunk count) by reading FAISS index metadata
- Main chat area:
  - `st.session_state.chat_history` — list of `ChatMessage`
  - Standard `st.chat_message` / `st.chat_input` loop
  - On new user input: show spinner "Thinking...", call `run_rag_pipeline`, append assistant
    response
  - Under each assistant response, an `st.expander("Sources")` showing:
    - The refined query used
    - Each retrieved chunk's source URL + a text snippet + similarity score
  - Handle errors gracefully (e.g., FAISS index not built yet → prompt user to click "Rebuild Index")

### `.env.example`
```
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

### `requirements.txt`
```
langgraph
langchain
langchain-community
langchain-openai
faiss-cpu
pydantic>=2
streamlit
python-dotenv
beautifulsoup4
```

### `tests/`
- `test_ingestion.py`: mock `WebBaseLoader` output, assert splitter produces chunks with correct
  metadata and chunk_size/overlap behavior
- `test_vectorstore.py`: build a tiny in-memory FAISS index from fake docs, assert MMR retriever
  returns diverse, non-duplicate results
- `test_graph.py`: mock the chat model calls, run the compiled graph end-to-end with a sample
  question, assert `GraphState` fields are populated in the correct order and `RAGResponse` is valid

### `README.md`
Include:
1. Project overview + architecture diagram description (text form, matching the pipeline above)
2. Setup steps: clone, `pip install -r requirements.txt`, copy `.env.example` → `.env`, add API key
3. `python -m ingestion.build_index` to build the FAISS index first
4. `streamlit run app.py` to launch
5. Notes on how to add more source URLs later (`SOURCE_URLS` in config + rebuild index)
6. Notes on extension points: hybrid search (BM25 + semantic), re-ranking (Cohere/cross-encoder),
   swap `RecursiveCharacterTextSplitter` for `SemanticChunker`

---

## BEHAVIORAL / QUALITY REQUIREMENTS

- All LLM calls must use structured, typed inputs/outputs where possible (Pydantic models), not raw
  dict manipulation.
- The answer generation prompt must enforce grounding and refuse to hallucinate outside retrieved
  context — this is policy/legal content, accuracy matters.
- Every answer must show which source URL(s) it drew from.
- Code should be modular so hybrid search + re-ranking can be added later without restructuring
  (this maps to your resume bullet about "hybrid semantic search with re-ranking" — build the
  current version with clean seams for that upgrade).
- No deployment config needed yet (no Docker/cloud) — but keep `config.py` env-driven so it's
  deployment-ready later.
- Add basic logging (`logging` module) at each graph node so you can trace refine → retrieve →
  context → answer during development.

---

## DELIVERABLE

A working local Streamlit app where:
1. Running `python -m ingestion.build_index` builds and saves a FAISS index from the two Stripe URLs.
2. Running `streamlit run app.py` opens a chat UI.
3. Asking a question like "How does Stripe use my personal data for fraud prevention?" returns a
   grounded answer citing the privacy policy page, visible in an expandable sources panel.
4. The full pipeline (refine → retrieve via MMR → build context → generate) runs as a LangGraph graph,
   not as ad-hoc function calls.
