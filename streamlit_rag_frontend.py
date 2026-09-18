import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)


# =========================== Page Config ===========================

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================== Dark Gradient Theme ===========================

THEME_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-deep: #0a0913;
    --bg-panel: #121022;
    --bg-card: #16142a;
    --border-soft: rgba(255,255,255,0.08);
    --text-primary: #F3F1F8;
    --text-muted: #9C97AE;
    --violet: #8B5CF6;
    --pink: #EC4899;
    --gradient: linear-gradient(90deg, #7C3AED 0%, #C026D3 55%, #EC4899 100%);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background with soft glowing blobs */
.stApp {
    background-color: var(--bg-deep);
    background-image:
        radial-gradient(circle at 15% 20%, rgba(124,58,237,0.20) 0%, transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(236,72,153,0.16) 0%, transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(139,92,246,0.14) 0%, transparent 50%);
    color: var(--text-primary);
}

.block-container {
    max-width: 880px;
    padding-top: 1.2rem;
    padding-bottom: 7rem;
}

/* ---------- Header banner ---------- */
.app-header {
    background: var(--gradient);
    border-radius: 18px;
    padding: 18px 26px;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 30px rgba(124,58,237,0.35);
}
.app-header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.app-header-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: rgba(255,255,255,0.18);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.1rem;
    color: white;
    backdrop-filter: blur(4px);
}
.app-header-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: white;
    line-height: 1.1;
}
.app-header-sub {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.85);
    margin-top: 2px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: var(--bg-panel);
    border-right: 1px solid var(--border-soft);
    min-width: 260px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary);
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] p {
    color: var(--text-muted) !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
}
.sidebar-brand-badge {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: var(--gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    color: white;
    font-size: 0.95rem;
}
.sidebar-brand-text {
    font-weight: 700;
    font-size: 1.02rem;
}

/* Buttons */
.stButton > button {
    background-color: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 0.5rem 0.85rem;
    font-weight: 500;
    transition: all 0.15s ease;
    box-shadow: none;
}
.stButton > button:hover {
    border-color: var(--violet);
    color: white;
    background: linear-gradient(90deg, rgba(124,58,237,0.25), rgba(236,72,153,0.2));
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
}

/* Primary gradient "New Chat" button */
.new-chat-btn .stButton > button {
    background: var(--gradient) !important;
    color: white !important;
    border: none !important;
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(192,38,211,0.35);
}
.new-chat-btn .stButton > button:hover {
    filter: brightness(1.08);
}

/* Thread id chip */
.thread-chip {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 0.74rem;
    color: var(--text-muted);
    font-family: monospace;
    margin-bottom: 1rem;
    word-break: break-all;
}

/* ---------- Empty state: centered hero (title + input) ---------- */
.st-key-center-wrap {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: center;
    min-height: 88vh;
    gap: 0.9rem;
}
.center-title-wrap {
    text-align: center;
    margin-bottom: 0.6rem;
}
.center-title-icon {
    width: 52px;
    height: 52px;
    margin: 0 auto 0.9rem auto;
    border-radius: 14px;
    background: var(--gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    color: white;
    box-shadow: 0 6px 22px rgba(192,38,211,0.4);
}
.center-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
}
.center-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Inline (non-pinned) chat_input used inside the centered container */
.st-key-center-wrap [data-testid="stChatInput"] {
    position: static !important;
}

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(236,72,153,0.10));
    border-color: rgba(139,92,246,0.35);
}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {
    border-top: none;
    background: transparent;
}
[data-testid="stChatInput"] > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}
[data-testid="stChatInput"] button {
    background: var(--gradient) !important;
    border-radius: 50% !important;
}
[data-testid="stChatInputSubmitButton"] svg,
[data-testid="stChatInput"] button svg {
    color: white !important;
    fill: white !important;
}

/* Suggested prompt chips */
.suggest-label {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-bottom: 0.6rem;
}
div[data-testid="column"] .stButton > button {
    border-radius: 999px;
    font-size: 0.82rem;
    padding: 0.45rem 1rem;
    background-color: var(--bg-card);
    border: 1px solid var(--border-soft);
}

/* Status / tool call boxes */
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
    border: 1px solid var(--border-soft) !important;
    background-color: var(--bg-card) !important;
}

/* Alerts */
.stAlert {
    border-radius: 10px;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-soft) !important;
}

/* Divider */
hr { border-color: var(--border-soft) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--border-soft); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

.footer-hint {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.76rem;
    margin-top: 0.6rem;
}

</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# =========================== Utilities ===========================

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def ingest_new_pdf(file_obj, thread_key, thread_docs):
    """Ingest a single uploaded PDF-like object if not already processed."""
    if file_obj.name in thread_docs:
        return False

    with st.status(f"Indexing `{file_obj.name}`...", expanded=True) as status_box:
        summary = ingest_pdf(
            file_obj.getvalue(),
            thread_id=thread_key,
            filename=file_obj.name,
        )
        thread_docs[file_obj.name] = summary
        status_box.update(
            label=f"✅ `{file_obj.name}` indexed",
            state="complete",
            expanded=False,
        )
    return True


def run_chat_turn(query_text, thread_key):
    """Send a query through the LangGraph chatbot and stream the response."""

    st.session_state["message_history"].append(
        {"role": "user", "content": query_text}
    )

    with st.chat_message("user"):
        st.write(query_text)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):

        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=query_text)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`...",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`...",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    if message_chunk.content:
                        yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}


add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])

thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

threads = st.session_state["chat_threads"][::-1]

selected_thread = None


# ============================ Sidebar ============================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-badge">OK</div>
            <div class="sidebar-brand-text">AI Chatbot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="thread-chip">Thread · {thread_key}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True):
        reset_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Current document status
    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.success(
            f"📄 `{latest_doc.get('filename')}`  \n"
            f"{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} pages"
        )
    else:
        st.info("No PDF indexed yet — attach one from the chat box below.")

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    st.subheader("Past conversations")

    if not threads:
        st.caption("No past conversations yet.")
    else:
        for tid in threads:
            label = f"💬  {str(tid)[:8]}…"
            if st.button(label, key=f"side-thread-{tid}", use_container_width=True):
                selected_thread = tid


# ============================ Chat History / Empty State =========

has_history = len(st.session_state["message_history"]) > 0

suggested_prompt = None
chat_submission = None

# Requires Streamlit >= 1.40 for the built-in `accept_file` attach button
# inside the chat input box. If your Streamlit is older, upgrade with:
#   pip install -U streamlit

if not has_history:
    # ---- Empty state: title + chat box centered in the middle of the screen ----
    with st.container(key="center-wrap"):

        st.markdown(
            """
            <div class="center-title-wrap">
                <div class="center-title-icon">✦</div>
                <div class="center-title">AI Chatbot</div>
                <div class="center-subtitle">Ask anything about your document</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chat_submission = st.chat_input(
            "Ask anything...",
            accept_file="multiple",
            file_type=["pdf"],
            key="center-chat-input",
        )

        st.markdown('<div class="suggest-label">✨ Try asking</div>', unsafe_allow_html=True)

        row1 = st.columns(2)
        row2 = st.columns(2)

        prompts = [
            "Summarize the uploaded document",
            "List the key takeaways",
            "What questions does this document answer?",
            "Explain this document in simple terms",
        ]

        for col, prompt in zip(row1 + row2, prompts):
            with col:
                if st.button(prompt, use_container_width=True, key=f"suggest-{prompt}"):
                    suggested_prompt = prompt

        st.markdown(
            '<div class="footer-hint">Click + to attach a PDF · hover messages to copy</div>',
            unsafe_allow_html=True,
        )

else:
    # ---- Normal chat mode: history on top, input pinned to the bottom ----
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    chat_submission = st.chat_input(
        "Ask anything...",
        accept_file="multiple",
        file_type=["pdf"],
        key="bottom-chat-input",
    )


query_text = None

if chat_submission:
    # Newer Streamlit returns an object with `.text` and `.files`
    submitted_text = getattr(chat_submission, "text", None) or (
        chat_submission if isinstance(chat_submission, str) else None
    )
    submitted_files = getattr(chat_submission, "files", None) or []

    for f in submitted_files:
        ingest_new_pdf(f, thread_key, thread_docs)

    if submitted_text:
        query_text = submitted_text
    elif submitted_files:
        query_text = "I've attached a PDF — please summarize it."

elif suggested_prompt:
    query_text = suggested_prompt


if query_text:
    run_chat_turn(query_text, thread_key)


# ============================ Document Info ======================

doc_meta = thread_document_metadata(thread_key)

if doc_meta:
    st.caption(
        f"📄 Document indexed: "
        f"**{doc_meta.get('filename')}** "
        f"(chunks: {doc_meta.get('chunks')}, "
        f"pages: {doc_meta.get('documents')})"
    )


# ============================ Load Old Chat ======================

st.divider()

if selected_thread:

    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(selected_thread)

    temp_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            continue

        temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages

    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})

    st.rerun()