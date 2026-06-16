import streamlit as st
import requests
import time
import os
import subprocess
import threading
import time

def start_api():
    subprocess.Popen(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])
    
threading.Thread(target=start_api, daemon=True).start()
time.sleep(3)
# ── Config ──────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .chat-message { padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; }
    .user-message { background-color: #1e3a5f; color: white; }
    .assistant-message { background-color: #1e1e2e; color: #e0e0e0; border-left: 3px solid #4a9eff; }
    .source-tag { background: #2a2a3e; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; color: #888; margin-right: 4px; }
    .latency-tag { color: #666; font-size: 0.75rem; }
    .stButton > button { width: 100%; }
    h1 { color: #4a9eff; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "avg_latency" not in st.session_state:
    st.session_state.avg_latency = []

# ── Helpers ──────────────────────────────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def upload_document(file_bytes, filename):
    try:
        r = requests.post(
            f"{API_URL}/documents/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=60
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return None

def query_rag(question, chat_history):
    """Query with chat history context injected into question"""
    try:
        # Build context from last 3 exchanges
        context = ""
        if chat_history:
            recent = chat_history[-3:]
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"
            question_with_context = f"{question}\n\n(Context from previous turns: {context})"
        else:
            question_with_context = question

        r = requests.post(
            f"{API_URL}/query",
            json={"question": question_with_context},
            timeout=30
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        return None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Q&A")
    st.caption("Powered by Vertex AI (Gemini) + pgvector")

    # API Status
    api_ok = check_api_health()
    if api_ok:
        st.success("API Connected ✅")
    else:
        st.error("API Offline ❌")
        st.caption(f"Expected at: {API_URL}")

    st.divider()

    # Document Upload
    st.subheader("📄 Upload Documents")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to query"
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    result = upload_document(f.read(), f.name)
                    if result:
                        st.session_state.uploaded_files.append(f.name)
                        st.success(f"✅ {f.name} ({result.get('chunks_processed', '?')} chunks)")
                    else:
                        st.error(f"❌ Failed to upload {f.name}")

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📚 Loaded Documents")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- {fname}")

    st.divider()

    # Stats
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        avg = round(sum(st.session_state.avg_latency) / len(st.session_state.avg_latency), 0) if st.session_state.avg_latency else 0
        st.metric("Avg Latency", f"{avg}ms")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.session_state.avg_latency = []
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("RAG Document Q&A Assistant")
st.caption("Upload PDFs in the sidebar, then ask questions below. Chat memory is maintained across turns.")

# Example questions
if not st.session_state.chat_history:
    st.info("👋 Upload a PDF document in the sidebar to get started, then ask any question about it.")
    st.markdown("**Example questions:**")
    cols = st.columns(3)
    examples = [
        "What is the main topic of this document?",
        "Summarize the key findings",
        "What are the conclusions?"
    ]
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state.pending_question = ex
                st.rerun()

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                for s in msg["sources"]:
                    fname = s.split("/")[-1]
                    sources_html += f'<span class="source-tag">📄 {fname}</span>'
            latency_html = f'<span class="latency-tag"> · {msg.get("latency_ms", "")}ms</span>' if msg.get("latency_ms") else ""
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Assistant:</strong> {msg['content']}
                <br><br>{sources_html}{latency_html}
            </div>
            """, unsafe_allow_html=True)

# Input
st.divider()
col_input, col_btn = st.columns([5, 1])

with col_input:
    question = st.text_input(
        "Ask a question",
        placeholder="Type your question here...",
        label_visibility="collapsed",
        key="question_input",
        value=st.session_state.get("pending_question", "")
    )

with col_btn:
    ask_clicked = st.button("Ask ➤", type="primary")

# Clear pending question
if "pending_question" in st.session_state:
    del st.session_state.pending_question

# Handle query
if ask_clicked and question.strip():
    if not api_ok:
        st.error("API is offline. Please start the FastAPI server first.")
    elif not st.session_state.uploaded_files:
        st.warning("Please upload at least one PDF document first.")
    else:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Thinking..."):
            result = query_rag(question, st.session_state.chat_history[:-1])

        if result:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "latency_ms": result.get("latency_ms", 0)
            })
            st.session_state.total_queries += 1
            st.session_state.avg_latency.append(result.get("latency_ms", 0))
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Sorry, I couldn't get an answer. Please check if the API is running.",
                "sources": [],
                "latency_ms": 0
            })

        st.rerun()