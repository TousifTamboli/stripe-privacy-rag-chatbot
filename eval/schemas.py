from typing import Literal, Optional
from pydantic import BaseModel, Field
from models.schemas import RetrievedChunk


class GoldenQAItem(BaseModel):
    id: str
    question: str
    gold_answer: str
    gold_source_url: str
    gold_chunk_ids: Optional[list[str]] = None
    category: Literal["factual", "procedural", "edge_case", "out_of_scope"]


class RetrievalScore(BaseModel):
    hit: bool
    recall_at_k: float
    reciprocal_rank: float
    context_precision: float


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    passed: bool = Field(description="Whether the criteria passed or met expectations")
    reasoning: str = Field(description="Brief explanation of the verdict and any detected issues")


class GenerationScore(BaseModel):
    faithfulness: JudgeVerdict
    answer_relevance: JudgeVerdict
    correctness: JudgeVerdict
    citation_correct: bool
    correctly_abstained: Optional[bool] = None


class EvalRecord(BaseModel):
    id: str
    question: str
    category: str
    refined_prompt: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk]
    retrieval_score: RetrievalScore
    generation_score: GenerationScore
    latency_ms: dict[str, float]
    timestamp: str
    passed: bool


class EvalReport(BaseModel):
    timestamp: str
    total_questions: int
    avg_hit_rate: float
    avg_recall_at_k: float
    avg_mrr: float
    avg_context_precision: float
    avg_faithfulness: float
    avg_relevance: float
    avg_correctness: float
    citation_accuracy: float
    avg_latency_ms: float
    records: list[EvalRecord]
    failures: list[EvalRecord]
