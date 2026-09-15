from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from models.schemas import ChatMessage, RAGResponse
from rag.graph import run_rag_pipeline
from rag.nodes import set_active_retriever


def test_rag_pipeline_end_to_end():
    # Setup mock retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Stripe uses transaction data to prevent fraud across its network.",
            metadata={
                "chunk_id": "test_chunk_1",
                "source_url": "https://stripe.com/in/privacy",
                "title": "Stripe Privacy Policy",
                "score": 0.95,
            },
        ),
        Document(
            page_content="Stripe retains legal information as required by financial regulations.",
            metadata={
                "chunk_id": "test_chunk_2",
                "source_url": "https://stripe.com/in/legal/privacy-center",
                "title": "Privacy Center",
                "score": 0.88,
            },
        ),
    ]
    set_active_retriever(mock_retriever)

    # Mock ChatOpenAI invoke
    def mock_invoke(messages):
        # If this is query refinement prompt
        prompt_content = str(messages)
        if "standalone" in prompt_content:
            return AIMessage(content="How does Stripe utilize personal information for fraud prevention?")
        else:
            return AIMessage(
                content="Stripe uses transaction data to prevent fraud across its network [Source: https://stripe.com/in/privacy]."
            )

    with patch("rag.nodes.ChatOpenAI") as mock_chat_cls:
        mock_chat_instance = MagicMock()
        mock_chat_instance.invoke.side_effect = mock_invoke
        mock_chat_cls.return_value = mock_chat_instance

        response = run_rag_pipeline(
            user_prompt="How do they stop fraud?",
            chat_history=[
                ChatMessage(role="user", content="Tell me about Stripe."),
                ChatMessage(role="assistant", content="Stripe is a financial infrastructure platform."),
            ],
        )

        assert isinstance(response, RAGResponse)
        assert "prevent fraud" in response.answer
        assert len(response.sources) >= 1
        assert "https://stripe.com/in/privacy" in response.sources
        assert len(response.retrieved_chunks) == 2
        assert response.refined_prompt == "How does Stripe utilize personal information for fraud prevention?"
