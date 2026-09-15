import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional
import pandas as pd
from tqdm import tqdm

from config import settings
from eval.schemas import (
    GoldenQAItem,
    RetrievalScore,
    GenerationScore,
    EvalRecord,
    EvalReport,
    JudgeVerdict,
)
from eval.golden_dataset import load_golden_dataset
from eval.retrieval_metrics import (
    hit_rate,
    recall_at_k,
    reciprocal_rank,
    context_precision,
)
from eval.judges import (
    judge_faithfulness,
    judge_answer_relevance,
    judge_correctness,
    judge_abstention,
    check_citation,
)
from rag.graph import rag_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_pipeline(
    dataset: Optional[list[GoldenQAItem]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> EvalReport:
    """Run full evaluation suite over the golden dataset and return an EvalReport."""
    items = dataset or load_golden_dataset()
    logger.info(f"Starting evaluation across {len(items)} items...")

    records: list[EvalRecord] = []
    failures: list[EvalRecord] = []

    for idx, item in enumerate(tqdm(items, desc="Evaluating RAG")):
        if progress_callback:
            progress_callback(idx, len(items), f"Evaluating: {item.question[:50]}...")

        # 1. Run pipeline and measure timing
        t0 = time.perf_counter()
        initial_state = {
            "user_prompt": item.question,
            "chat_history": [],
            "refined_prompt": "",
            "retrieved_chunks": [],
            "context": "",
            "answer": "",
            "sources": [],
        }

        final_state = rag_graph.invoke(initial_state)
        total_latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_chunks = final_state.get("retrieved_chunks", [])
        answer = final_state.get("answer", "")
        refined_prompt = final_state.get("refined_prompt", item.question)
        context = final_state.get("context", "")
        sources = final_state.get("sources", [])

        retrieved_ids = [c.metadata.chunk_id for c in retrieved_chunks]
        retrieved_sources = [c.metadata.source_url for c in retrieved_chunks]

        # 2. Score Retrieval
        is_hit = hit_rate(retrieved_chunks, item.gold_source_url)
        r_at_k = recall_at_k(retrieved_ids, item.gold_chunk_ids, hit_fallback=is_hit)
        mrr = reciprocal_rank(
            retrieved_ids,
            item.gold_chunk_ids,
            retrieved_sources=retrieved_sources,
            gold_source_url=item.gold_source_url,
        )
        c_prec = context_precision(item.question, retrieved_chunks)

        retrieval_score = RetrievalScore(
            hit=is_hit,
            recall_at_k=r_at_k,
            reciprocal_rank=mrr,
            context_precision=c_prec,
        )

        # 3. Score Generation
        if item.category == "out_of_scope":
            abstain_verdict = judge_abstention(item.question, answer)
            relevance_verdict = judge_answer_relevance(item.question, answer)
            # For out of scope, if abstained, faithfulness is 1.0, correctness is 1.0
            faithfulness_verdict = JudgeVerdict(
                score=abstain_verdict.score,
                passed=abstain_verdict.passed,
                reasoning=f"Abstention check: {abstain_verdict.reasoning}",
            )
            correctness_verdict = JudgeVerdict(
                score=abstain_verdict.score,
                passed=abstain_verdict.passed,
                reasoning=f"Out of scope correctly answered via abstention: {abstain_verdict.reasoning}",
            )
            citation_ok = check_citation(answer, item.gold_source_url, category="out_of_scope")

            generation_score = GenerationScore(
                faithfulness=faithfulness_verdict,
                answer_relevance=relevance_verdict,
                correctness=correctness_verdict,
                citation_correct=citation_ok,
                correctly_abstained=abstain_verdict.passed,
            )
            record_passed = abstain_verdict.passed
        else:
            faithfulness_verdict = judge_faithfulness(answer, context)
            relevance_verdict = judge_answer_relevance(item.question, answer)
            correctness_verdict = judge_correctness(answer, item.gold_answer, category=item.category)
            citation_ok = check_citation(answer, item.gold_source_url, category=item.category)

            generation_score = GenerationScore(
                faithfulness=faithfulness_verdict,
                answer_relevance=relevance_verdict,
                correctness=correctness_verdict,
                citation_correct=citation_ok,
                correctly_abstained=None,
            )

            # Record passes if all primary thresholds are met
            record_passed = (
                is_hit
                and (faithfulness_verdict.score >= settings.MIN_FAITHFULNESS)
                and (relevance_verdict.score >= settings.MIN_ANSWER_RELEVANCE)
                and (correctness_verdict.score >= settings.MIN_CORRECTNESS)
            )

        record = EvalRecord(
            id=item.id,
            question=item.question,
            category=item.category,
            refined_prompt=refined_prompt,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            retrieval_score=retrieval_score,
            generation_score=generation_score,
            latency_ms={"total": round(total_latency_ms, 2)},
            timestamp=datetime.now(timezone.utc).isoformat(),
            passed=record_passed,
        )

        records.append(record)
        if not record_passed:
            failures.append(record)

    # 4. Aggregations
    total_q = len(records)
    avg_hit = sum(1.0 for r in records if r.retrieval_score.hit) / total_q if total_q else 0.0
    avg_recall = sum(r.retrieval_score.recall_at_k for r in records) / total_q if total_q else 0.0
    avg_mrr = sum(r.retrieval_score.reciprocal_rank for r in records) / total_q if total_q else 0.0
    avg_c_prec = sum(r.retrieval_score.context_precision for r in records) / total_q if total_q else 0.0
    avg_faith = sum(r.generation_score.faithfulness.score for r in records) / total_q if total_q else 0.0
    avg_rel = sum(r.generation_score.answer_relevance.score for r in records) / total_q if total_q else 0.0
    avg_corr = sum(r.generation_score.correctness.score for r in records) / total_q if total_q else 0.0
    avg_cit = sum(1.0 for r in records if r.generation_score.citation_correct) / total_q if total_q else 0.0
    avg_lat = sum(r.latency_ms.get("total", 0.0) for r in records) / total_q if total_q else 0.0

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = EvalReport(
        timestamp=timestamp_str,
        total_questions=total_q,
        avg_hit_rate=round(avg_hit, 3),
        avg_recall_at_k=round(avg_recall, 3),
        avg_mrr=round(avg_mrr, 3),
        avg_context_precision=round(avg_c_prec, 3),
        avg_faithfulness=round(avg_faith, 3),
        avg_relevance=round(avg_rel, 3),
        avg_correctness=round(avg_corr, 3),
        citation_accuracy=round(avg_cit, 3),
        avg_latency_ms=round(avg_lat, 2),
        records=records,
        failures=failures,
    )

    # 5. Persist Report JSON
    os.makedirs(settings.EVAL_RESULTS_PATH, exist_ok=True)
    report_file = os.path.join(settings.EVAL_RESULTS_PATH, f"eval_{timestamp_str}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation report written to: {report_file}")

    return report


def print_console_summary(report: EvalReport):
    """Print a clean Pandas summary table and failure details to console."""
    print("\n======================= EVALUATION SUMMARY =======================")
    print(f"Timestamp:             {report.timestamp}")
    print(f"Total Questions:       {report.total_questions}")
    print(f"Overall Passed:        {report.total_questions - len(report.failures)} / {report.total_questions}")
    print(f"Hit Rate @ k:          {report.avg_hit_rate:.1%}")
    print(f"Faithfulness:          {report.avg_faithfulness:.1%}")
    print(f"Answer Relevance:      {report.avg_relevance:.1%}")
    print(f"Correctness:           {report.avg_correctness:.1%}")
    print(f"Citation Accuracy:     {report.citation_accuracy:.1%}")
    print(f"Avg Latency:           {report.avg_latency_ms:.0f} ms")
    print("===================================================================\n")

    # Category breakdown table
    rows = []
    for r in report.records:
        rows.append({
            "Category": r.category,
            "Hit": 1 if r.retrieval_score.hit else 0,
            "Faithfulness": r.generation_score.faithfulness.score,
            "Relevance": r.generation_score.answer_relevance.score,
            "Correctness": r.generation_score.correctness.score,
            "Passed": 1 if r.passed else 0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        category_summary = df.groupby("Category").mean().round(3)
        print("--- Performance By Category ---")
        print(category_summary.to_string())
        print("\n")

    if report.failures:
        print(f"🚨 {len(report.failures)} QUESTIONS FLAGGED AS FAILURES / BELOW THRESHOLD:")
        for fail in report.failures:
            print(f"- [{fail.id}] ({fail.category}): {fail.question}")
            print(f"  Answer: {fail.answer[:120]}...")
            print(f"  Faithfulness: {fail.generation_score.faithfulness.score} | Correctness: {fail.generation_score.correctness.score}")
            print(f"  Reason: {fail.generation_score.faithfulness.reasoning or fail.generation_score.correctness.reasoning}\n")
    else:
        print("🎉 All test questions passed quality thresholds!")


if __name__ == "__main__":
    rep = evaluate_pipeline()
    print_console_summary(rep)
