from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    source_url: str
    chunk_id: str
    title: Optional[str] = None


class RetrievedChunk(BaseModel):
    content: str
    metadata: ChunkMetadata
    score: float = Field(default=0.0, description="Relevance or similarity score if available")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    refined_prompt: Optional[str] = None
