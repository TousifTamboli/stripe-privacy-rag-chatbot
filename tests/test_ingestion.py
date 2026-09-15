from unittest.mock import patch
from langchain_core.documents import Document
from ingestion.loader import load_documents
from ingestion.splitter import split_documents
from config import settings


def test_load_documents_mocked():
    test_urls = ["https://stripe.com/in/privacy"]
    mock_doc = Document(
        page_content="Stripe processes personal data to prevent fraud.",
        metadata={"source": "https://stripe.com/in/privacy"},
    )

    with patch("ingestion.loader.WebBaseLoader.load", return_value=[mock_doc]):
        docs = load_documents(test_urls)
        assert len(docs) == 1
        assert docs[0].page_content == "Stripe processes personal data to prevent fraud."
        assert docs[0].metadata["source_url"] == "https://stripe.com/in/privacy"
        assert "title" in docs[0].metadata


def test_split_documents():
    long_content = "Stripe Privacy Policy statement. " * 80  # ~2560 chars
    docs = [
        Document(
            page_content=long_content,
            metadata={"source_url": "https://stripe.com/in/privacy"},
        )
    ]

    chunks = split_documents(docs)
    assert len(chunks) > 1

    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["source_url"] == "https://stripe.com/in/privacy"
        assert len(chunk.page_content) <= settings.CHUNK_SIZE + 100  # allow minor boundary leeway
