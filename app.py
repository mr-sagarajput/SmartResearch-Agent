"""
Modern Streamlit Web Interface for SmartResearch Chatbot.
=========================================================
Dual-Tab Architecture with In-Line Chat:
- Tab 1: 💬 Smart Q&A & Document Chat (Mode 1: Zero PDF clutter, purely conversational)
- Tab 2: 🔬 Deep Autonomous Research (Mode 2: Researches topic, saves PDF in output/, shows summary & opens LIVE CHAT right there!)
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SmartResearch Chatbot AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
)
from core.agent_router import execute_deep_research, process_query_with_router
from core.llm_gateway import get_gemini_client, get_openrouter_client
from core.vector_engine import (
    build_document_chunks,
    compute_bytes_hash,
    embed_chunks,
    load_cached_embeddings,
    load_pdf_pages_from_bytes,
    save_cached_embeddings,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Messages specifically for Tab 2 research conversation
if "research_messages" not in st.session_state:
    st.session_state.research_messages = []

if "indexed_pdf_hash" not in st.session_state:
    st.session_state.indexed_pdf_hash = None

if "indexed_pdf_name" not in st.session_state:
    st.session_state.indexed_pdf_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

if "active_research_topic" not in st.session_state:
    st.session_state.active_research_topic = None

if "active_research_pdf" not in st.session_state:
    st.session_state.active_research_pdf = None

if "active_research_summary" not in st.session_state:
    st.session_state.active_research_summary = None


gemini_client = get_gemini_client()
openrouter_client = get_openrouter_client()

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Sidebar Interface
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 SmartResearch AI")
    st.caption("Agentic-RAG: Vector Memory + Autonomous Web Research")
    st.markdown("---")

    # API Status Badges
    c1, c2 = st.columns(2)
    with c1:
        if OPENROUTER_API_KEY:
            st.success("OpenRouter\nReady", icon="🟢")
        else:
            st.error("OpenRouter\nMissing", icon="🔴")
    with c2:
        if GEMINI_API_KEY:
            st.success("Gemini\nReady", icon="🟢")
        else:
            st.error("Gemini\nMissing", icon="🔴")

    st.markdown("### 📥 Index Document (Optional)")
    st.caption(f"Limit: **Max {MAX_FILE_SIZE_MB} MB per PDF**")
    uploaded_file = st.file_uploader(
        "Upload a PDF to ground conversations",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None and gemini_client is not None:
        file_bytes = uploaded_file.getvalue()
        file_size_mb = len(file_bytes) / (1024 * 1024)

        # 5 MB Strict File Size Guard
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            st.error(
                f"❌ **File Too Large!** Uploaded PDF is **{file_size_mb:.2f} MB**.\n\n"
                f"The maximum allowed size is **{MAX_FILE_SIZE_MB} MB** to ensure fast processing and zero RAM strain.",
                icon="⚠️",
            )
        else:
            file_hash = compute_bytes_hash(file_bytes)

            if st.session_state.indexed_pdf_hash != file_hash:
                cached_data = load_cached_embeddings(file_hash)

                if cached_data is not None:
                    chunks, chunk_embeddings = cached_data
                    st.session_state.chunks = chunks
                    st.session_state.chunk_embeddings = chunk_embeddings
                    st.session_state.indexed_pdf_hash = file_hash
                    st.session_state.indexed_pdf_name = uploaded_file.name
                    st.session_state.messages = []
                    st.toast(f"⚡ Loaded {len(chunks)} chunks instantly from disk cache!", icon="🚀")
                    st.rerun()
                else:
                    status_box = st.status(f"Indexing '{uploaded_file.name}'...", expanded=True)
                    try:
                        pages = load_pdf_pages_from_bytes(
                            file_bytes,
                            gemini_client,
                            status_callback=lambda msg: status_box.write(msg),
                        )
                        chunks = build_document_chunks(pages)
                        status_box.write(f"✂️ Created {len(chunks)} semantic chunks. Embedding with Rate-Pacing...")

                        chunk_embeddings = embed_chunks(
                            gemini_client,
                            chunks,
                            progress_callback=lambda msg: status_box.write(f"  ↳ {msg}"),
                        )
                        save_cached_embeddings(file_hash, chunks, chunk_embeddings)

                        st.session_state.chunks = chunks
                        st.session_state.chunk_embeddings = chunk_embeddings
                        st.session_state.indexed_pdf_hash = file_hash
                        st.session_state.indexed_pdf_name = uploaded_file.name
                        st.session_state.messages = []

                        status_box.update(label="✅ Document Indexed & Cached Successfully!", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:
                        status_box.update(label="❌ Indexing failed", state="error", expanded=True)
                        st.error(f"Error: {e}")

    # Active Document Status
    if st.session_state.indexed_pdf_name:
        st.markdown("---")
        st.markdown("### 📊 Active Knowledge")
        st.info(
            f"**Doc:** `{st.session_state.indexed_pdf_name}`\n\n"
            f"**Total Chunks:** `{len(st.session_state.chunks)}`\n\n"
            f"**Web Fallback:** `🟢 Active`"
        )
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color: gray;'>Engine: Vector DB + DuckDuckGo + OpenRouter / Gemini</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Dual-Tab Navigation
# ---------------------------------------------------------------------------

tab_chat, tab_research = st.tabs(["💬 Smart Q&A & Chat (Mode 1)", "🔬 Deep Topic Research (Mode 2)"])

# ===========================================================================
# TAB 1: Mode 1 - Smart Q&A (No PDF Clutter)
# ===========================================================================
with tab_chat:
    st.header("💬 Smart Conversational Assistant", divider="rainbow")

    if not st.session_state.indexed_pdf_name and not st.session_state.messages:
        st.info("💡 **Welcome!** Upload a PDF from the sidebar, or ask any question directly. If information is missing from the document, I will search the live web automatically.")

    # Render History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "badge" in msg:
                st.caption(msg["badge"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Retrieved Sources ({len(msg['sources'])} references)"):
                    for item, score in msg["sources"]:
                        url_part = f" — [Link]({item.get('url')})" if item.get("url") else ""
                        st.markdown(f"**{item['source']}**{url_part}")
                        st.caption(item["text"][:250] + "...")
                        st.markdown("---")

    # Mode 1 Chat Input
    if user_question := st.chat_input("Ask a question about your document or any topic...", key="chat_input_tab1"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            def update_agent_status(msg):
                status_placeholder.info(f"🤖 {msg}")

            try:
                result = process_query_with_router(
                    user_query=user_question,
                    chunks=st.session_state.chunks,
                    chunk_embeddings=st.session_state.chunk_embeddings,
                    history=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]],
                    status_callback=update_agent_status,
                )

                status_placeholder.empty()

                if "updated_embeddings" in result and result["updated_embeddings"] is not None:
                    st.session_state.chunk_embeddings = result["updated_embeddings"]

                st.markdown(result["answer"])

                badge_text = ""
                if result["is_web_research"]:
                    badge_text = f"🌐 Answered via Live Web Research · {result['provider']}"
                    st.caption(badge_text)
                else:
                    badge_text = f"⚡ Answered from Vector Memory (Confidence: {result['max_score']:.1%}) · {result['provider']}"
                    st.caption(badge_text)

                if result["sources"]:
                    with st.expander(f"📚 View Sources ({len(result['sources'])} references)"):
                        for item, score in result["sources"]:
                            url_part = f" — [Link]({item.get('url')})" if item.get("url") else ""
                            st.markdown(f"**{item['source']}**{url_part}")
                            st.caption(item["text"][:250] + "...")
                            st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "badge": badge_text,
                    "sources": result["sources"],
                })

            except Exception as e:
                status_placeholder.empty()
                err_msg = f"❌ Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})


# ===========================================================================
# TAB 2: Mode 2 - Deep Autonomous Topic Research (Clean Badge + INLINE CHAT)
# ===========================================================================
with tab_research:
    st.header("🔬 Autonomous Topic Research & Live Discussion", divider="rainbow")
    st.markdown(
        "Enter any topic to conduct deep web research. The agent will crawl multiple live sources, "
        "save a verified PDF report to your `output/` folder, and **immediately open a live discussion below** "
        "so you can ask follow-up questions directly without switching tabs!"
    )

    col_topic, col_btn = st.columns([4, 1])
    with col_topic:
        research_input = st.text_input(
            "Topic to Research:",
            placeholder="e.g., Breakthroughs in Quantum Physics and Black Holes",
            label_visibility="collapsed",
            key="deep_research_query_box",
        )
    with col_btn:
        start_research = st.button("🚀 Run Research", use_container_width=True, type="primary")

    if start_research and research_input.strip():
        research_status = st.status(f"Conducting Deep Research on '{research_input}'...", expanded=True)
        try:
            res = execute_deep_research(
                topic=research_input.strip(),
                chunks=st.session_state.chunks,
                chunk_embeddings=st.session_state.chunk_embeddings,
                max_pages=4,
                status_callback=lambda m: research_status.write(m),
            )

            if res.get("success"):
                if res.get("updated_embeddings") is not None:
                    st.session_state.chunk_embeddings = res["updated_embeddings"]

                st.session_state.active_research_topic = research_input.strip()
                st.session_state.active_research_pdf = res["pdf_path"]
                st.session_state.active_research_summary = res["summary"]

                # Reset previous research messages for the new topic
                st.session_state.research_messages = [
                    {
                        "role": "assistant",
                        "content": f"### 📋 Executive Research Summary: {research_input.strip()}\n\n{res['summary']}",
                        "badge": f"📄 Saved directly to: `{res['pdf_path']}`",
                        "sources": [({"source": s["title"], "page": 0, "text": s["notes"], "url": s["url"]}, 1.0) for s in res["sources"]],
                    }
                ]

                research_status.update(label="✅ Deep Research Completed & PDF Saved!", state="complete", expanded=False)
                st.rerun()

            else:
                research_status.update(label="❌ Research Failed", state="error", expanded=True)
                st.error(f"Error: {res.get('error')}")

        except Exception as e:
            research_status.update(label="❌ Error during research", state="error", expanded=True)
            st.error(f"Error: {e}")

    # --- Active Research Display & Inline Chat Session ---
    if st.session_state.active_research_topic and st.session_state.research_messages:
        st.markdown("---")
        
        # Clean Path Badge (No redundant bulky download button)
        col_badge, col_clear = st.columns([5, 1])
        with col_badge:
            st.success(
                f"✅ **Report Saved:** `{st.session_state.active_research_pdf.name}` "
                f"*(Stored in `output/` folder)*",
                icon="📄",
            )
        with col_clear:
            if st.button("🔄 New Topic", use_container_width=True):
                st.session_state.active_research_topic = None
                st.session_state.active_research_pdf = None
                st.session_state.active_research_summary = None
                st.session_state.research_messages = []
                st.rerun()

        st.subheader(f"💬 Live Discussion: {st.session_state.active_research_topic}")

        # Render conversation history for this research report
        for r_msg in st.session_state.research_messages:
            with st.chat_message(r_msg["role"]):
                st.markdown(r_msg["content"])
                if "badge" in r_msg:
                    st.caption(r_msg["badge"])
                if "sources" in r_msg and r_msg["sources"]:
                    with st.expander(f"📚 View Research Citations ({len(r_msg['sources'])} sources)"):
                        for item, score in r_msg["sources"]:
                            url_part = f" — [Link]({item.get('url')})" if item.get("url") else ""
                            st.markdown(f"**{item['source']}**{url_part}")
                            st.caption(item["text"][:250] + "...")
                            st.markdown("---")

        # In-Line Chat Input directly in Mode 2!
        if follow_up_q := st.chat_input("Ask any question directly about this research report...", key="chat_input_tab2"):
            st.session_state.research_messages.append({"role": "user", "content": follow_up_q})
            with st.chat_message("user"):
                st.markdown(follow_up_q)

            with st.chat_message("assistant"):
                status_box_r = st.empty()

                def update_r_status(msg):
                    status_box_r.info(f"🤖 {msg}")

                try:
                    res_q = process_query_with_router(
                        user_query=follow_up_q,
                        chunks=st.session_state.chunks,
                        chunk_embeddings=st.session_state.chunk_embeddings,
                        history=[{"role": m["role"], "content": m["content"]} for m in st.session_state.research_messages[:-1]],
                        status_callback=update_r_status,
                    )
                    status_box_r.empty()

                    if "updated_embeddings" in res_q and res_q["updated_embeddings"] is not None:
                        st.session_state.chunk_embeddings = res_q["updated_embeddings"]

                    st.markdown(res_q["answer"])

                    badge_info = f"⚡ Grounded from Active Research Report (Confidence: {res_q['max_score']:.1%}) · {res_q['provider']}"
                    st.caption(badge_info)

                    if res_q["sources"]:
                        with st.expander(f"📚 View Citations ({len(res_q['sources'])} references)"):
                            for item, score in res_q["sources"]:
                                url_part = f" — [Link]({item.get('url')})" if item.get("url") else ""
                                st.markdown(f"**{item['source']}**{url_part}")
                                st.caption(item["text"][:250] + "...")
                                st.markdown("---")

                    st.session_state.research_messages.append({
                        "role": "assistant",
                        "content": res_q["answer"],
                        "badge": badge_info,
                        "sources": res_q["sources"],
                    })

                except Exception as ex:
                    status_box_r.empty()
                    err_txt = f"❌ Error: {ex}"
                    st.error(err_txt)
                    st.session_state.research_messages.append({"role": "assistant", "content": err_txt})
