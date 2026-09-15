import logging
from langgraph.graph import StateGraph, START, END

from models.schemas import ChatMessage, RAGResponse
from rag.state import GraphState
from rag.nodes import refine_query, retrieve, build_context, generate_answer

logger = logging.getLogger(__name__)


def create_rag_graph():
    """Build and compile the LangGraph StateGraph for the Stripe RAG pipeline."""
    workflow = StateGraph(GraphState)

    # 1. Add nodes
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("build_context", build_context)
    workflow.add_node("generate_answer", generate_answer)

    # 2. Add edges: START -> refine_query -> retrieve -> build_context -> generate_answer -> END
    workflow.add_edge(START, "refine_query")
    workflow.add_edge("refine_query", "retrieve")
    workflow.add_edge("retrieve", "build_context")
    workflow.add_edge("build_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


rag_graph = create_rag_graph()


def run_rag_pipeline(user_prompt: str, chat_history: list[ChatMessage] | None = None) -> RAGResponse:
    """Execute the full LangGraph RAG pipeline and return a typed RAGResponse.
    
    Args:
        user_prompt: The raw user prompt from the chat input.
        chat_history: Prior chat conversation history.
        
    Returns:
        RAGResponse: Grounded answer, source URLs, retrieved chunks, and refined query.
    """
    history = chat_history or []
    initial_state: GraphState = {
        "user_prompt": user_prompt,
        "chat_history": history,
        "refined_prompt": "",
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "sources": [],
    }

    logger.info(f"Invoking RAG pipeline for user prompt: '{user_prompt}'")
    final_state = rag_graph.invoke(initial_state)

    return RAGResponse(
        answer=final_state.get("answer", ""),
        sources=final_state.get("sources", []),
        retrieved_chunks=final_state.get("retrieved_chunks", []),
        refined_prompt=final_state.get("refined_prompt", user_prompt),
    )
