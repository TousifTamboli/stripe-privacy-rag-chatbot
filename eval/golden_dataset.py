import json
import os
import logging
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import settings
from eval.schemas import GoldenQAItem

logger = logging.getLogger(__name__)
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")


def load_golden_dataset(file_path: Optional[str] = None) -> list[GoldenQAItem]:
    """Load the golden evaluation dataset from JSON."""
    path = file_path or DEFAULT_DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden dataset file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [GoldenQAItem(**item) for item in data]
    logger.info(f"Loaded {len(items)} GoldenQAItem(s) from {path}")
    return items


def save_golden_dataset(items: list[GoldenQAItem], file_path: Optional[str] = None) -> None:
    """Save golden evaluation dataset items to a JSON file."""
    path = file_path or DEFAULT_DATASET_PATH
    data = [item.model_dump() for item in items]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(items)} items to {path}")


def generate_candidate_questions(
    chunk_text: str,
    source_url: str,
    num_questions: int = 3,
) -> list[dict]:
    """Helper utility: Use an LLM to propose candidate Q&A pairs from raw policy chunk text."""
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set to generate candidate questions.")

    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.3,
    )

    prompt = (
        f"You are helping curate a golden test dataset for a RAG chatbot on Stripe's policies.\n"
        f"Source URL: {source_url}\n"
        f"Chunk Text:\n{chunk_text}\n\n"
        f"Generate {num_questions} high quality Question-Answer pairs that can be answered strictly "
        f"from this text. For each pair, categorize it as 'factual', 'procedural', or 'edge_case'.\n"
        f"Format the output as a valid JSON array of objects with keys: question, gold_answer, category."
    )

    messages = [
        SystemMessage(content="You are a data curation assistant for legal/privacy RAG testing. Output only valid JSON."),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        logger.error(f"Failed to parse generated candidate questions: {e}")
        return []
