import os
import streamlit as st

from config import settings
from models.schemas import ChatMessage, RAGResponse
from rag.graph import run_rag_pipeline
from rag.vectorstore import load_vectorstore, get_retriever
from rag.nodes import set_active_retriever
from ingestion.build_index import build_index
from eval.report import render_streamlit_eval_tab


def get_index_status() -> dict:
    """Check FAISS index files on disk and return status info."""
    index_path = settings.FAISS_INDEX_PATH
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        try:
            vs = load_vectorstore()
            total_vectors = vs.index.ntotal if hasattr(vs, "index") else "Available"
            return {"exists": True, "count": total_vectors}
        except Exception:
            return {"exists": True, "count": "Unknown"}
    return {"exists": False, "count": 0}


st.set_page_config(
    page_title="Stripe Policy Assistant & Evaluation",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Stripe Policy Assistant")
st.caption("Grounded Q&A powered by LangGraph, FAISS MMR Retrieval, and Automated LLM Evaluation")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration & Index")

    status = get_index_status()
    if status["exists"]:
        st.success(f"● Index Ready ({status['count']} chunks indexed)")
    else:
        st.warning("⚠️ Index Not Built")

    st.subheader("Indexed Sources")
    for url in settings.SOURCE_URLS:
        st.markdown(f"- [{url}]({url})")

    st.markdown("---")
    if st.button("🔄 Rebuild Index", use_container_width=True):
        if not settings.OPENAI_API_KEY:
            st.error("Please configure OPENAI_API_KEY in your .env file before building the index.")
        else:
            with st.spinner("Scraping Stripe pages, chunking & creating FAISS index..."):
                try:
                    num_chunks = build_index()
                    new_vs = load_vectorstore()
                    set_active_retriever(get_retriever(new_vs))
                    st.success(f"Successfully rebuilt index with {num_chunks} chunks!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error rebuilding index: {e}")

    st.markdown("---")
    st.caption(
        f"**Models**\n"
        f"- Chat & Judge: `{settings.CHAT_MODEL}`\n"
        f"- Embeddings: `{settings.EMBEDDING_MODEL}`\n\n"
        f"**Retrieval**\n"
        f"- MMR Search (k={settings.RETRIEVER_K}, fetch_k={settings.RETRIEVER_FETCH_K})"
    )

tab_chat, tab_eval = st.tabs(["💬 Policy Chat", "📊 Evaluation Dashboard"])

with tab_chat:
    # Display prior chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            rag_obj = msg.get("response_obj")
            if rag_obj and rag_obj.retrieved_chunks:
                with st.expander("📚 Sources & Retrieval Trace"):
                    if rag_obj.refined_prompt:
                        st.markdown(f"**Refined Query:** `{rag_obj.refined_prompt}`")
                    
                    for i, chunk in enumerate(rag_obj.retrieved_chunks, 1):
                        st.markdown(f"**Chunk {i}** | [Source URL]({chunk.metadata.source_url})")
                        snippet = chunk.content.strip().replace("\n", " ")
                        st.text(f"{snippet[:280]}...")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about Stripe's privacy and legal policies..."):
        if not status["exists"]:
            st.error("FAISS index is not built yet! Please click 'Rebuild Index' in the sidebar to ingest Stripe policy documents.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt, "response_obj": None})
            with st.chat_message("user"):
                st.markdown(prompt)

            chat_history = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]

            with st.chat_message("assistant"):
                with st.spinner("Searching Stripe policies and formulating answer..."):
                    try:
                        rag_response: RAGResponse = run_rag_pipeline(
                            user_prompt=prompt,
                            chat_history=chat_history,
                        )
                        st.markdown(rag_response.answer)

                        if rag_response.retrieved_chunks:
                            with st.expander("📚 Sources & Retrieval Trace"):
                                if rag_response.refined_prompt:
                                    st.markdown(f"**Refined Query:** `{rag_response.refined_prompt}`")
                                
                                for i, chunk in enumerate(rag_response.retrieved_chunks, 1):
                                    st.markdown(f"**Chunk {i}** | [Source URL]({chunk.metadata.source_url})")
                                    snippet = chunk.content.strip().replace("\n", " ")
                                    st.text(f"{snippet[:280]}...")
                                    st.divider()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": rag_response.answer,
                            "response_obj": rag_response,
                        })

                    except Exception as e:
                        error_msg = f"An error occurred while processing your request: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "response_obj": None,
                        })

with tab_eval:
    render_streamlit_eval_tab()
