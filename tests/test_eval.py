from unittest.mock import MagicMock, patch
from models.schemas import RetrievedChunk, ChunkMetadata, RAGResponse
from eval.schemas import GoldenQAItem, JudgeVerdict
from eval.retrieval_metrics import (
    hit_rate,
    recall_at_k,
    reciprocal_rank,
    context_precision,
)
from eval.judges import check_citation
from eval.run_eval import evaluate_pipeline


def test_retrieval_metrics_accuracy():
    chunks = [
        RetrievedChunk(
            content="Stripe protects user privacy.",
            metadata=ChunkMetadata(source_url="https://stripe.com/in/privacy", chunk_id="c1"),
        ),
        RetrievedChunk(
            content="Stripe maintains PCI compliance.",
            metadata=ChunkMetadata(source_url="https://stripe.com/in/legal/privacy-center", chunk_id="c2"),
        ),
    ]

    # Hit rate
    assert hit_rate(chunks, "https://stripe.com/in/privacy") is True
    assert hit_rate(chunks, "https://stripe.com/in/other-page") is False

    # Recall at k
    assert recall_at_k(["c1", "c2"], ["c1", "c3"]) == 0.5
    assert recall_at_k(["c1", "c2"], None, hit_fallback=True) == 1.0

    # Reciprocal Rank
    assert reciprocal_rank(["c1", "c2"], ["c2"]) == 0.5
    assert reciprocal_rank(["c1", "c2"], ["c1"]) == 1.0
    assert reciprocal_rank(["c1", "c2"], ["c99"]) == 0.0

    # Context precision
    assert context_precision("privacy question", chunks) >= 0.5


def test_check_citation():
    answer_with_cite = "According to [Source: https://stripe.com/in/privacy], Stripe encrypts data."
    assert check_citation(answer_with_cite, "https://stripe.com/in/privacy", category="factual") is True
    assert check_citation(answer_with_cite, "https://stripe.com/in/legal/privacy-center", category="factual") is False
    # Out of scope should pass citation check by design
    assert check_citation("I do not have this information.", "https://stripe.com/in/privacy", category="out_of_scope") is True


@patch("eval.run_eval.rag_graph.invoke")
@patch("eval.run_eval.judge_faithfulness")
@patch("eval.run_eval.judge_answer_relevance")
@patch("eval.run_eval.judge_correctness")
@patch("eval.run_eval.judge_abstention")
def test_evaluate_pipeline_aggregation(
    mock_abstain,
    mock_correct,
    mock_relevance,
    mock_faith,
    mock_invoke,
):
    # Setup test dataset with 1 factual and 1 out_of_scope
    test_dataset = [
        GoldenQAItem(
            id="test_1",
            question="What data does Stripe collect?",
            gold_answer="Identity and financial data.",
            gold_source_url="https://stripe.com/in/privacy",
            gold_chunk_ids=["chunk_1"],
            category="factual",
        ),
        GoldenQAItem(
            id="test_2",
            question="What is Stripe Atlas price?",
            gold_answer="I don't have that information.",
            gold_source_url="https://stripe.com/in/privacy",
            gold_chunk_ids=None,
            category="out_of_scope",
        ),
    ]

    mock_invoke.return_value = {
        "user_prompt": "test",
        "refined_prompt": "test refined",
        "retrieved_chunks": [
            RetrievedChunk(
                content="Stripe collects identity data.",
                metadata=ChunkMetadata(source_url="https://stripe.com/in/privacy", chunk_id="chunk_1"),
            )
        ],
        "context": "Stripe collects identity data.",
        "answer": "Stripe collects identity data [Source: https://stripe.com/in/privacy].",
        "sources": ["https://stripe.com/in/privacy"],
    }

    mock_faith.return_value = JudgeVerdict(score=1.0, passed=True, reasoning="Fully grounded.")
    mock_relevance.return_value = JudgeVerdict(score=1.0, passed=True, reasoning="Directly answered.")
    mock_correct.return_value = JudgeVerdict(score=1.0, passed=True, reasoning="Matches gold answer.")
    mock_abstain.return_value = JudgeVerdict(score=1.0, passed=True, reasoning="Properly declined.")

    report = evaluate_pipeline(dataset=test_dataset)

    assert report.total_questions == 2
    assert report.avg_hit_rate == 1.0
    assert report.avg_faithfulness == 1.0
    assert report.avg_relevance == 1.0
    assert report.avg_correctness == 1.0
    assert len(report.failures) == 0
    assert report.records[1].generation_score.correctly_abstained is True
