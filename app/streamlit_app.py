import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv
from loader import PDFLoader
from chunker import TextChunker
from retriever import Retriever
from generator import Generator

load_dotenv(dotenv_path="../.env")

st.set_page_config(
    page_title="Medical RAG Assistant",
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def get_core_services():
    """
    Instantiate retriever and generator once per session.
    Cached so they survive reruns but reset on full restart.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in .env file.")
        st.stop()
    return Retriever(), Generator(api_key=api_key)


def index_pdf(uploaded_file, retriever: Retriever) -> dict:
    """Save upload to a temp file, index it, return stats."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PDFLoader(tmp_path)
        docs = loader.load()
        chunker = TextChunker()
        chunks = chunker.split(docs)

        retriever.reset()
        retriever.index_chunks(chunks)

        return {
            "pages": len(docs),
            "chunks": len(chunks),
            "filename": uploaded_file.name
        }
    finally:
        os.unlink(tmp_path)


def render_sources(results: list):
    with st.expander(f"Sources retrieved ({len(results)})", expanded=False):
        for i, r in enumerate(results, 1):
            page   = r["metadata"].get("page", "?")
            score  = r.get("score", 0)
            source = r.get("source", "?")
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(
                    f"**Source {i}**  \n"
                    f"Page `{page}`  \n"
                    f"Score `{score:.3f}`  \n"
                    f"via `{source}`"
                )
            with col2:
                st.text_area(
                    label=f"chunk_{i}",
                    value=r["content"],
                    height=100,
                    disabled=True,
                    label_visibility="collapsed"
                )
            if i < len(results):
                st.divider()


def main():
    retriever, generator = get_core_services()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.title("Medical RAG Assistant")
        st.caption("Upload a medical guidelines PDF to begin.")
        st.divider()

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type="pdf",
            help="Upload any medical guidelines PDF. Previous document is replaced on new upload."
        )

        if uploaded_file:
            if uploaded_file.size > 50 * 1024 * 1024:
                st.error(
                    f"File too large: {uploaded_file.size / (1024 * 1024):.1f} MB. "
                    "Please upload a PDF under 50 MB."
                )
                st.stop()
            last_file = st.session_state.get("indexed_file")

            if last_file != uploaded_file.name:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    stats = index_pdf(uploaded_file, retriever)

                st.session_state.indexed_file = uploaded_file.name
                st.session_state.doc_stats    = stats
                st.session_state.history      = []  # clear chat on new doc
                st.rerun()

        if st.session_state.get("doc_stats"):
            stats = st.session_state.doc_stats
            st.success(f"Ready: **{stats['filename']}**")
            st.markdown(
                f"- {stats['pages']} pages  \n"
                f"- {stats['chunks']} chunks indexed"
            )
            st.divider()

            if st.button("Clear document", use_container_width=True):
                retriever.reset()
                st.session_state.clear()
                st.rerun()

    # ── Main area ────────────────────────────────────────────
    st.title("Medical RAG Assistant")

    if not st.session_state.get("indexed_file"):
        st.info("Upload a PDF in the sidebar to get started.")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Render previous turns
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            render_sources(turn["results"])

    # New query
    query = st.chat_input("Ask a question about the uploaded document...")

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving sources..."):
                results = retriever.retrieve(query, top_k=5)

            with st.spinner("Generating answer..."):
                answer = generator.get_answer(query, results)

            st.markdown(answer)
            render_sources(results)

        st.session_state.history.append({
            "query": query,
            "answer": answer,
            "results": results
        })


if __name__ == "__main__":
    main()