import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from config import settings
from models.schemas import RetrievedChunk, ChunkMetadata
from rag.prompts import QUERY_REFINEMENT_SYSTEM_PROMPT, ANSWER_SYSTEM_PROMPT
from rag.state import GraphState
from rag.vectorstore import load_vectorstore, get_retriever

logger = logging.getLogger(__name__)

# Cache retriever instance across graph runs
_retriever = None


def get_active_retriever():
    """Retrieve the cached retriever instance or initialize one."""
    global _retriever
    if _retriever is None:
        vs = load_vectorstore()
        _retriever = get_retriever(vs)
    return _retriever


def set_active_retriever(retriever):
    """Set or override the active retriever (useful for testing or index reload)."""
    global _retriever
    _retriever = retriever


def refine_query(state: GraphState) -> dict:
    """Node 1: Rewrites and clarifies the user prompt using conversation history."""
    logger.info("--- NODE: refine_query ---")
    user_prompt = state.get("user_prompt", "")
    chat_history = state.get("chat_history", [])

    messages = [SystemMessage(content=QUERY_REFINEMENT_SYSTEM_PROMPT)]

    # Append past conversation context
    for msg in chat_history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    messages.append(
        HumanMessage(
            content=f"Conversation history is above. Please rewrite the following user question into a standalone retrieval query:\n\n{user_prompt}"
        )
    )

    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0,
    )

    try:
        response = llm.invoke(messages)
        refined_query = response.content.strip()
    except Exception as e:
        logger.warning(f"Query refinement LLM call failed ({e}). Falling back to raw prompt.")
        refined_query = user_prompt

    logger.info(f"Refined prompt: '{refined_query}'")
    return {"refined_prompt": refined_query}


def retrieve(state: GraphState) -> dict:
    """Node 2: Retrieves relevant policy documents via MMR search."""
    logger.info("--- NODE: retrieve ---")
    query = state.get("refined_prompt") or state.get("user_prompt", "")
    retriever = get_active_retriever()

    docs = retriever.invoke(query)
    logger.info(f"Retriever returned {len(docs)} documents for query: '{query}'")

    retrieved_chunks: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()

    for idx, doc in enumerate(docs):
        metadata_dict = doc.metadata or {}
        chunk_id = metadata_dict.get("chunk_id", f"retrieved_{idx}")

        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        chunk_meta = ChunkMetadata(
            source_url=metadata_dict.get("source_url", "https://stripe.com/in/privacy"),
            chunk_id=chunk_id,
            title=metadata_dict.get("title"),
        )

        retrieved_chunks.append(
            RetrievedChunk(
                content=doc.page_content,
                metadata=chunk_meta,
                score=float(metadata_dict.get("score", 0.0)),
            )
        )

    return {"retrieved_chunks": retrieved_chunks}


def build_context(state: GraphState) -> dict:
    """Node 3: Concatenates retrieved chunks with source markers within a token/char budget."""
    logger.info("--- NODE: build_context ---")
    chunks = state.get("retrieved_chunks", [])
    max_char_budget = 12000  # Safe context window for gpt-4o-mini

    context_parts: list[str] = []
    sources: list[str] = []
    current_length = 0

    for chunk in chunks:
        source_url = chunk.metadata.source_url
        if source_url not in sources:
            sources.append(source_url)

        formatted_part = f"[Source: {source_url}]\n{chunk.content}\n"
        part_len = len(formatted_part)

        if current_length + part_len > max_char_budget:
            remaining = max_char_budget - current_length
            if remaining > 100:
                context_parts.append(formatted_part[:remaining] + "\n...[truncated]")
            break

        context_parts.append(formatted_part)
        current_length += part_len

    context = "\n---\n".join(context_parts)
    logger.info(f"Built context of {len(context)} characters across {len(sources)} unique source(s).")
    return {"context": context, "sources": sources}


def generate_answer(state: GraphState) -> dict:
    """Node 4: Generates a grounded response with source citations."""
    logger.info("--- NODE: generate_answer ---")
    context = state.get("context", "")
    query = state.get("refined_prompt") or state.get("user_prompt", "")

    if not context.strip():
        return {
            "answer": "I don't have that information in the indexed Stripe policy documentation.",
            "sources": [],
        }

    system_content = (
        f"{ANSWER_SYSTEM_PROMPT}\n\n"
        f"CONTEXT INFORMATION:\n"
        f"{context}\n"
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"User Question: {query}"),
    ]

    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
    )

    response = llm.invoke(messages)
    answer = response.content.strip()
    logger.info("Generated grounded answer successfully.")
    return {"answer": answer}
