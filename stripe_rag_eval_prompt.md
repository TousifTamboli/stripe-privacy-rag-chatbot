# Build Prompt: Evaluation Suite for the Stripe RAG Chatbot

Copy everything below into Claude Code (or use as your own spec) to add a full evaluation layer
on top of the existing `stripe-rag` project (LangGraph + LangChain + Pydantic + FAISS + OpenAI +
Streamlit).

---

## GOAL

Build an evaluation suite that measures both **retrieval quality** (did we fetch the right chunks?)
and **generation quality** (is the answer faithful, relevant, and correct?) for the Stripe policy
RAG pipeline. Use an LLM-as-judge approach with strict Pydantic-typed outputs so scores are
structured, reproducible, and stay consistent with the rest of the stack (no new frameworks like
RAGAS required, though it's noted as an optional swap-in). Produce a golden Q&A dataset from the
two indexed Stripe pages, run the full pipeline against it, score every stage, and output a report
(JSON + a human-readable Streamlit/markdown summary).

---

## WHAT TO EVALUATE (map to the architecture)

1. **Retrieval stage** (`retrieve` node / MMR retriever over FAISS)
   - Hit Rate @k — did the retrieved chunks contain the source that actually answers the question?
   - Recall @k — of all "gold" relevant chunks for a question, how many were retrieved?
   - MRR (Mean Reciprocal Rank) — how high up was the first relevant chunk?
   - Context Precision — of the chunks retrieved, what fraction were actually relevant (via LLM judge)?

2. **Query refinement stage** (`refine_query` node)
   - Refinement quality — does the rewritten query preserve intent and improve retrievability?
     (LLM judge, pass/fail + short reason)

3. **Generation stage** (`generate_answer` node)
   - **Faithfulness / Groundedness** — is every claim in the answer supported by the retrieved
     context? (critical for policy/legal content — must catch hallucination)
   - **Answer Relevance** — does the answer actually address the user's question?
   - **Correctness** — compare model answer to a human-written/gold reference answer (semantic
     similarity + LLM judge verdict)
   - **Citation Accuracy** — does the answer cite the correct source URL(s)?

4. **End-to-end**
   - Latency per stage (refine / retrieve / generate) and total
   - Token usage / approximate cost per query

---

## TECH STACK ADDITIONS

- Reuse: `langgraph`, `langchain`, `langchain-openai`, `pydantic`, `faiss-cpu`, `openai`
- New: `pandas` (for results tables), `tqdm` (progress bars), optionally `ragas` + `datasets`
  as a **commented-out alternative** eval backend (note in code but don't hard-depend on it)
- No new frontend framework — reuse Streamlit for a results dashboard tab

---

## PROJECT STRUCTURE ADDITIONS

```
stripe-rag/
├── eval/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models for eval records, scores, reports
│   ├── golden_dataset.py       # Generates/stores the Q&A golden dataset
│   ├── golden_dataset.json     # Persisted golden dataset (source_url, question, gold_answer, gold_chunk_ids)
│   ["stripe_rag_build_prompt.md"] # existing file, unchanged
│   ├── judges.py                # LLM-as-judge functions (faithfulness, relevance, correctness, refinement quality)
│   ├── retrieval_metrics.py     # hit rate, recall@k, MRR, context precision
│   ├── run_eval.py              # CLI: runs full pipeline against golden dataset, scores everything
│   └── report.py                # Aggregates results -> markdown/JSON report + Streamlit tab renderer
├── data/
│   └── eval_results/            # timestamped JSON + markdown reports (gitignored)
```

---

## DETAILED REQUIREMENTS BY FILE

### `eval/schemas.py`
Pydantic models:
- `GoldenQAItem(BaseModel)`:
  - `id: str`
  - `question: str`
  - `gold_answer: str` — human-written reference answer
  - `gold_source_url: str` — which of the two pages should answer this
  - `gold_chunk_ids: list[str] | None` — optional, filled in after first index build by manually
    tagging which chunk(s) contain the answer (or auto-matched via string containment as a fallback)
  - `category: Literal["factual","procedural","edge_case","out_of_scope"]` — see dataset section below

- `RetrievalScore(BaseModel)`: `hit: bool`, `recall_at_k: float`, `reciprocal_rank: float`,
  `context_precision: float`

- `JudgeVerdict(BaseModel)`: `score: float` (0–1), `passed: bool`, `reasoning: str`
  — this is the **structured output** every LLM-judge call must return (use
  `.with_structured_output(JudgeVerdict)` in LangChain/OpenAI function-calling mode, not free text
  parsing)

- `GenerationScore(BaseModel)`: `faithfulness: JudgeVerdict`, `answer_relevance: JudgeVerdict`,
  `correctness: JudgeVerdict`, `citation_correct: bool`

- `EvalRecord(BaseModel)`: full record per question — `question`, `refined_prompt`, `answer`,
  `retrieved_chunks: list[RetrievedChunk]`, `retrieval_score: RetrievalScore`,
  `generation_score: GenerationScore`, `latency_ms: dict[str, float]`, `timestamp: str`

- `EvalReport(BaseModel)`: aggregated — `total_questions`, `avg_hit_rate`, `avg_recall_at_k`,
  `avg_mrr`, `avg_faithfulness`, `avg_relevance`, `avg_correctness`, `citation_accuracy`,
  `avg_latency_ms`, `records: list[EvalRecord]`, `failures: list[EvalRecord]` (below-threshold items)

### `eval/golden_dataset.py`
- Build **~20–30 Q&A pairs** across both Stripe pages, spanning these categories:
  - **factual** (10–12): direct lookup questions, e.g. "What categories of personal data does
    Stripe collect?", "How long does Stripe retain personal data?"
  - **procedural** (4–6): "How can a user request deletion of their data from Stripe?", "How do I
    opt out of Stripe's data sharing?"
  - **edge_case** (4–6): ambiguous/compound questions, e.g. "Does Stripe sell my data to
    advertisers and can I stop it?" (tests whether refinement + retrieval handles multi-part asks),
    or questions where the phrasing doesn't match policy vocabulary (tests semantic search, not
    keyword match)
  - **out_of_scope** (3–4): questions NOT answerable from these two pages, e.g. "What are Stripe's
    card processing fees?" — the correct behavior is the bot saying it doesn't have that info,
    NOT hallucinating. This directly tests the faithfulness/grounding requirement.
- Each item authored manually (best quality) with `gold_answer` and `gold_source_url` filled in by
  reading the actual pages. Provide a `generate_candidate_questions()` helper that uses an LLM to
  *propose* candidate questions from the indexed chunks (to speed up authoring), but the final
  `golden_dataset.json` should be human-reviewed/edited before being treated as ground truth.
- Persist as `eval/golden_dataset.json`, loaded via `load_golden_dataset() -> list[GoldenQAItem]`.

### `eval/retrieval_metrics.py`
- `hit_rate(retrieved_chunks, gold_source_url) -> bool` — True if any retrieved chunk's
  `metadata.source_url == gold_source_url`
- `recall_at_k(retrieved_chunk_ids, gold_chunk_ids) -> float`
- `reciprocal_rank(retrieved_chunk_ids, gold_chunk_ids) -> float`
- `context_precision(question, retrieved_chunks, judge_fn) -> float` — for each retrieved chunk,
  ask the LLM judge "is this chunk relevant to answering the question?" (binary), average the
  results

### `eval/judges.py`
All judges use a chat model (`gpt-4o-mini` or a stronger model like `gpt-4o` since judging needs
higher reliability than generation) with `.with_structured_output(JudgeVerdict)`:

- `judge_faithfulness(answer, context) -> JudgeVerdict` — prompt: "Given the CONTEXT and the
  ANSWER, identify if every factual claim in the answer is directly supported by the context.
  Score 1.0 if fully grounded, 0.0 if any claim is unsupported or hallucinated. List unsupported
  claims in reasoning."
- `judge_answer_relevance(question, answer) -> JudgeVerdict` — does the answer address what was
  asked, regardless of factual correctness?
- `judge_correctness(answer, gold_answer) -> JudgeVerdict` — semantic comparison against the human
  reference; score partial credit for partially correct answers, explain gaps in reasoning
- `judge_refinement_quality(original_question, refined_prompt) -> JudgeVerdict` — does the refined
  prompt preserve the original intent and read as a good standalone retrieval query?
- `check_citation(answer, gold_source_url) -> bool` — simple string-containment check that the
  gold URL (or its domain path) appears in the answer's cited sources; flag out_of_scope items
  specially (expected behavior = NO confident citation, an "I don't know" response is a PASS not
  a FAIL)

Use low `temperature` (0 or 0.1) for all judges to keep scoring consistent across runs.

### `eval/run_eval.py`
CLI script (`python -m eval.run_eval`):
1. Load golden dataset
2. Load the compiled `rag_graph` from `rag/graph.py`
3. For each `GoldenQAItem`:
   - Run `run_rag_pipeline(question, chat_history=[])`, capturing per-node latency (wrap each node
     call with a timer, or use LangGraph's built-in run metadata if available)
   - Compute `RetrievalScore` via `retrieval_metrics.py`
   - Compute `GenerationScore` via `judges.py`
   - Assemble an `EvalRecord`
4. Handle `out_of_scope` items specially: success = model correctly declines / expresses
   uncertainty rather than fabricating an answer — score this as a boolean
   `correctly_abstained` field, don't penalize low "correctness" score against a gold answer that
   doesn't exist
5. Aggregate into `EvalReport`, save to `data/eval_results/eval_{timestamp}.json`
6. Print a summary table (via `pandas.DataFrame`) to console: per-category breakdown of hit rate,
   faithfulness, relevance, correctness
7. Flag and print any record scoring below thresholds (configurable, defaults below) as "failures"
   for manual review

**Default pass thresholds** (add to `config.py`):
```python
MIN_HIT_RATE = 0.8
MIN_FAITHFULNESS = 0.9   # highest bar — hallucination on policy content is the worst failure mode
MIN_ANSWER_RELEVANCE = 0.8
MIN_CORRECTNESS = 0.7
```

### `eval/report.py`
- `render_markdown_report(report: EvalReport) -> str` — writes a clean markdown summary: overall
  scores, per-category table, list of failing questions with reasoning excerpts, latency stats
- `render_streamlit_eval_tab(report: EvalReport)` — add a second tab/page to the existing Streamlit
  app ("Evaluation Dashboard") that:
  - Shows aggregate metrics as `st.metric` cards (hit rate, faithfulness, relevance, correctness,
    avg latency)
  - Shows a bar chart of scores by category (factual / procedural / edge_case / out_of_scope)
  - Shows a filterable table of all `EvalRecord`s with expandable rows for full judge reasoning
  - "Re-run Evaluation" button that triggers `run_eval.py` logic live with a progress bar

### `tests/test_eval.py`
- Test `retrieval_metrics.py` functions with hand-crafted chunk lists (known hit/miss cases)
- Mock LLM judge calls (fixed `JudgeVerdict` returns) and assert `run_eval.py` correctly aggregates
  scores and flags failures below threshold
- Assert `out_of_scope` items are scored via the abstention path, not the correctness path

---

## BEHAVIORAL / QUALITY REQUIREMENTS

- Every judge call must return a **typed Pydantic object**, never free-text parsed with regex —
  use OpenAI structured outputs / function calling under the hood.
- Faithfulness is the most important metric given the legal/policy domain — set its threshold
  highest and treat failures here as blocking, not just "nice to fix."
- The eval suite must be able to run standalone (`python -m eval.run_eval`) without touching the
  Streamlit app, and also be visualized inside Streamlit — don't couple the scoring logic to the UI.
- Keep the golden dataset version-controlled (`golden_dataset.json` in git) so eval runs are
  comparable across code changes over time — this lets you track regressions when you later add
  hybrid search / re-ranking.
- Log every eval run with a timestamp so you can diff `eval_{timestamp1}.json` vs
  `eval_{timestamp2}.json` to see if a pipeline change improved or regressed scores.
- Note in `README.md` under a new "Evaluation" section: how to add new golden questions, how to
  re-run eval, how to interpret the thresholds, and how this suite would extend to RAGAS or a
  cross-encoder re-ranker eval later.

---

## DELIVERABLE

1. `eval/golden_dataset.json` with ~20–30 hand-reviewed Q&A pairs across factual / procedural /
   edge_case / out_of_scope categories, sourced from the two Stripe pages.
2. `python -m eval.run_eval` runs the full pipeline against the dataset and prints/saves a scored
   report covering retrieval (hit rate, recall@k, MRR, context precision) and generation
   (faithfulness, relevance, correctness, citation accuracy), plus latency.
3. A "correctly abstained" check for out-of-scope questions to explicitly catch hallucination.
4. A Streamlit "Evaluation Dashboard" tab visualizing the latest report with per-category
   breakdowns and drill-down into individual failing questions.
5. All scoring logic implemented as reusable, typed Pydantic functions — not a one-off script — so
   it can be re-run every time the pipeline changes (new splitter, re-ranker, different retriever
   settings, etc.) to track quality over time.
