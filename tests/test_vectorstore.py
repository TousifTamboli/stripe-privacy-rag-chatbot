import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from rag.vectorstore import get_retriever, load_vectorstore


def test_mmr_retriever_diversity():
    fake_embeddings = FakeEmbeddings(size=32)
    docs = [
        Document(page_content="Stripe uses cookies for analytics.", metadata={"chunk_id": "c1", "source_url": "https://stripe.com/1"}),
        Document(page_content="Stripe uses cookies for session management.", metadata={"chunk_id": "c2", "source_url": "https://stripe.com/1"}),
        Document(page_content="Stripe protects financial data with encryption.", metadata={"chunk_id": "c3", "source_url": "https://stripe.com/2"}),
        Document(page_content="Fraud detection involves IP analysis and machine learning.", metadata={"chunk_id": "c4", "source_url": "https://stripe.com/2"}),
    ]

    vectorstore = FAISS.from_documents(docs, fake_embeddings)
    retriever = get_retriever(vectorstore)

    results = retriever.invoke("How does Stripe handle cookies?")
    assert len(results) > 0
    # Ensure no duplicates in returned results
    ids = [d.metadata.get("chunk_id") for d in results]
    assert len(ids) == len(set(ids))


def test_load_vectorstore_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_vectorstore("/non/existent/path/for/testing")
