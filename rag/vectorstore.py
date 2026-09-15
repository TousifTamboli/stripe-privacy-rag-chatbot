import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config import settings

logger = logging.getLogger(__name__)


def load_vectorstore(index_path: str | None = None) -> FAISS:
    """Load the persisted FAISS vector store from disk.
    
    Raises:
        FileNotFoundError: If index files do not exist at the target path.
    """
    path = index_path or settings.FAISS_INDEX_PATH
    faiss_file = os.path.join(path, "index.faiss")
    pkl_file = os.path.join(path, "index.pkl")

    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'. "
            "Please build the index first by running `python -m ingestion.build_index` "
            "or clicking 'Rebuild Index' in the Streamlit sidebar."
        )

    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )

    logger.info(f"Loading FAISS index from {path}...")
    vectorstore = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def get_retriever(vectorstore: FAISS) -> VectorStoreRetriever:
    """Create and configure an MMR (Maximal Marginal Relevance) retriever from FAISS vectorstore."""
    logger.info(
        f"Configuring MMR retriever: k={settings.RETRIEVER_K}, "
        f"fetch_k={settings.RETRIEVER_FETCH_K}, lambda_mult={settings.MMR_LAMBDA}"
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.RETRIEVER_K,
            "fetch_k": settings.RETRIEVER_FETCH_K,
            "lambda_mult": settings.MMR_LAMBDA,
        },
    )
