from typing import TypedDict
from models.schemas import ChatMessage, RetrievedChunk


class GraphState(TypedDict):
    user_prompt: str
    chat_history: list[ChatMessage]
    refined_prompt: str
    retrieved_chunks: list[RetrievedChunk]
    context: str
    answer: str
    sources: list[str]
