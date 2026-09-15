import uuid
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)

# Alternative chunker for future exploration:
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai import OpenAIEmbeddings
# def split_documents_semantic(docs: list[Document]) -> list[Document]:
#     semantic_splitter = SemanticChunker(
#         OpenAIEmbeddings(model=settings.EMBEDDING_MODEL),
#         breakpoint_threshold_type="percentile"
#     )
#     return semantic_splitter.split_documents(docs)


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter and inject chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} document(s) into {len(chunks)} chunk(s)")

    for idx, chunk in enumerate(chunks):
        source_url = chunk.metadata.get("source_url", "unknown")
        # Ensure unique chunk_id per chunk
        chunk_unique_id = f"chunk_{idx}_{uuid.uuid4().hex[:8]}"
        chunk.metadata["chunk_id"] = chunk_unique_id
        chunk.metadata["source_url"] = source_url
        if "title" not in chunk.metadata:
            chunk.metadata["title"] = "Stripe Policy Document"

    return chunks
