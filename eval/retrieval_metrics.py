from typing import Callable, Optional
from models.schemas import RetrievedChunk


def hit_rate(retrieved_chunks: list[RetrievedChunk], gold_source_url: str) -> bool:
    """Check if any retrieved chunk came from the expected gold source URL."""
    if not retrieved_chunks:
        return False
    gold_clean = gold_source_url.rstrip("/").lower()
    return any(
        chunk.metadata.source_url.rstrip("/").lower() == gold_clean
        for chunk in retrieved_chunks
    )


def recall_at_k(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: Optional[list[str]] = None,
    hit_fallback: bool = False,
) -> float:
    """Compute Recall@k.
    
    If gold_chunk_ids is provided, computes fraction of gold chunks retrieved.
    Otherwise falls back to hit_fallback (1.0 if hit, 0.0 otherwise).
    """
    if not gold_chunk_ids:
        return 1.0 if hit_fallback else 0.0

    retrieved_set = set(retrieved_chunk_ids)
    matched = sum(1 for cid in gold_chunk_ids if cid in retrieved_set)
    return float(matched / len(gold_chunk_ids))


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: Optional[list[str]] = None,
    retrieved_sources: Optional[list[str]] = None,
    gold_source_url: Optional[str] = None,
) -> float:
    """Compute Reciprocal Rank (1 / rank of first relevant item)."""
    # 1. Exact chunk match if gold chunk IDs exist
    if gold_chunk_ids:
        gold_set = set(gold_chunk_ids)
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in gold_set:
                return 1.0 / rank
        return 0.0

    # 2. Source URL match fallback
    if retrieved_sources and gold_source_url:
        gold_clean = gold_source_url.rstrip("/").lower()
        for rank, src in enumerate(retrieved_sources, start=1):
            if src.rstrip("/").lower() == gold_clean:
                return 1.0 / rank

    return 0.0


def context_precision(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    chunk_relevance_judge_fn: Optional[Callable[[str, str], bool]] = None,
) -> float:
    """Compute context precision as the ratio of relevant chunks to total retrieved chunks.
    
    Uses chunk_relevance_judge_fn if provided; otherwise estimates via term/source alignment.
    """
    if not retrieved_chunks:
        return 0.0

    if chunk_relevance_judge_fn:
        relevant_count = sum(
            1 for chunk in retrieved_chunks if chunk_relevance_judge_fn(question, chunk.content)
        )
        return float(relevant_count / len(retrieved_chunks))

    # Basic heuristic when no LLM judge is supplied for individual chunks
    # (Checking non-trivial word overlap)
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    relevant_count = 0
    for chunk in retrieved_chunks:
        c_text = chunk.content.lower()
        if any(w in c_text for w in q_words):
            relevant_count += 1

    return float(relevant_count / len(retrieved_chunks))
