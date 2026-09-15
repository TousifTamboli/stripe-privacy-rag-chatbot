import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import settings
from ingestion.loader import load_documents
from ingestion.splitter import split_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_index(source_urls: list[str] | None = None) -> int:
    """Build and persist the FAISS vector index from specified or default source URLs.
    
    Returns:
        int: Number of chunks indexed.
    """
    urls = source_urls or settings.SOURCE_URLS
    logger.info(f"Starting index build for {len(urls)} source URL(s)...")

    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please set it in your .env file or environment variables."
        )

    # 1. Load documents
    docs = load_documents(urls)
    if not docs:
        raise RuntimeError("No documents were loaded from the source URLs.")

    # 2. Split documents into chunks
    chunks = split_documents(docs)
    if not chunks:
        raise RuntimeError("Text splitting produced 0 chunks.")

    logger.info(f"Successfully generated {len(chunks)} chunks.")

    # 3. Create embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )

    # 4. Build FAISS index
    logger.info("Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Persist to disk
    os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(settings.FAISS_INDEX_PATH)
    logger.info(f"FAISS index successfully saved to: {settings.FAISS_INDEX_PATH}")

    # 6. Sanity check: print chunk count and sample chunk
    print(f"\n✅ Index Build Complete!")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Index persisted at: {settings.FAISS_INDEX_PATH}")
    if chunks:
        sample = chunks[0]
        preview = sample.page_content[:200].replace("\n", " ")
        print(f"\n--- Sample Chunk Preview ---")
        print(f"Source URL: {sample.metadata.get('source_url')}")
        print(f"Chunk ID:   {sample.metadata.get('chunk_id')}")
        print(f"Content:    {preview}...\n")

    return len(chunks)


if __name__ == "__main__":
    build_index()
