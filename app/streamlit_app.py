import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.chat-message { padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; }
.user-message { background-color: #1e3a5f; color: white; }
.assistant-message { background-color: #1e1e2e; color: #e0e0e0; border-left: 3px solid #4a9eff; }
.source-tag { background: #2a2a3e; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; color: #888; margin-right: 4px; }
.latency-tag { color: #666; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Init RAG pipeline once ───────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_groq import ChatGroq

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )
    return embeddings, vectorstore, llm

# ── Session state ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "latencies" not in st.session_state:
    st.session_state.latencies = []

# ── Load pipeline ────────────────────────────────────────────────────────────
with st.spinner("Loading AI pipeline..."):
    try:
        embeddings, vectorstore, llm = load_pipeline()
        pipeline_ok = True
    except Exception as e:
        pipeline_ok = False
        pipeline_error = str(e)

# ── Helpers ──────────────────────────────────────────────────────────────────
def ingest_pdf(file_bytes, filename):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(file_bytes)
    tmp.close()

    loader = PyPDFLoader(tmp.name)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    metadatas = [{"source": filename} for _ in chunks]
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    os.unlink(tmp.name)
    return len(chunks)

def query_rag(question):
    start = time.time()
    results = vectorstore.similarity_search(question, k=3)
    if not results:
        return "No relevant documents found. Please upload a PDF first.", [], 0

    context = "\n\n".join([f"[Source: {r.metadata.get('source','unknown')}]\n{r.page_content}" for r in results])
    sources = list(set([r.metadata.get('source','unknown') for r in results]))

    prompt = f"""Based on the following documents, answer the question concisely and accurately.
If the answer is not in the documents, say so clearly.

Documents:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    latency = round((time.time() - start) * 1000, 1)
    return response.content, sources, latency

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Q&A")
    st.caption("Powered by Groq (Llama 3.3) + ChromaDB")

    if pipeline_ok:
        st.success("Pipeline Ready ✅")
    else:
        st.error(f"Pipeline Error ❌\n{pipeline_error}")

    st.divider()
    st.subheader("📄 Upload Documents")
    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded and pipeline_ok:
        for f in uploaded:
            if f.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    try:
                        chunks = ingest_pdf(f.read(), f.name)
                        st.session_state.uploaded_files.append(f.name)
                        st.success(f"✅ {f.name} ({chunks} chunks)")
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)}")

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📚 Loaded Documents")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- {fname}")

    st.divider()
    st.subheader("📊 Session Stats")
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.total_queries)
    avg = round(sum(st.session_state.latencies)/len(st.session_state.latencies)) if st.session_state.latencies else 0
    c2.metric("Avg Latency", f"{avg}ms")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.session_state.latencies = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("RAG Document Q&A Assistant")
st.caption("Upload PDFs in the sidebar, then ask questions below. Chat memory maintained across turns.")

if not st.session_state.chat_history:
    st.info("👋 Upload a PDF document in the sidebar to get started.")
    cols = st.columns(3)
    for i, ex in enumerate(["What is the main topic?", "Summarize key findings", "What are the conclusions?"]):
        if cols[i].button(ex, key=f"ex{i}"):
            st.session_state["pending"] = ex
            st.rerun()

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        src_html = "".join([f'<span class="source-tag">📄 {s.split("/")[-1]}</span>' for s in msg.get("sources",[])])
        lat_html = f'<span class="latency-tag"> · {msg.get("latency_ms","")}ms</span>'
        st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {msg["content"]}<br><br>{src_html}{lat_html}</div>', unsafe_allow_html=True)

st.divider()
col1, col2 = st.columns([5,1])
with col1:
    question = st.text_input("Ask", placeholder="Type your question here...", label_visibility="collapsed",
                              value=st.session_state.pop("pending", ""), key="q_input")
with col2:
    ask = st.button("Ask ➤", type="primary")

if ask and question.strip():
    if not pipeline_ok:
        st.error("Pipeline not loaded.")
    elif not st.session_state.uploaded_files:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer, sources, latency = query_rag(question)
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources, "latency_ms": latency})
        st.session_state.total_queries += 1
        st.session_state.latencies.append(latency)
        st.rerun()