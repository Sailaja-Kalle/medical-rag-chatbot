import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import streamlit as st
import glob
from pdf_loader import load_pdfs
from chunking import split_documents
from embeddings import get_embedding_model
from rag_pipeline import vector_db_exists
from retriever import get_retriever
from chatbot import load_llm
from langchain_community.vectorstores import Chroma
from utils import load_history, save_message, clear_history

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #74ebd5, #ACB6E5); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #16213e) !important; }
section[data-testid="stSidebar"] label { color: #38ef7d !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] p { color: white !important; }
section[data-testid="stSidebar"] li { color: white !important; }
section[data-testid="stSidebar"] h2 { color: #38ef7d !important; }
section[data-testid="stSidebar"] h3 { color: #38ef7d !important; }
section[data-testid="stSidebar"] span { color: white !important; }
section[data-testid="stSidebar"] .stButton button { background: linear-gradient(90deg, #11998e, #38ef7d) !important; color: #000 !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: linear-gradient(135deg, #11998e, #302b63) !important; border: 2px dashed #38ef7d !important; border-radius: 8px !important; color: white !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: white !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background: #38ef7d !important; color: #000 !important; font-weight: 700 !important; border: none !important; border-radius: 6px !important; }
.chat-header { background: linear-gradient(90deg, #1a1a2e, #302b63); padding: 2rem; border-radius: 16px; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
.chat-header h1 { color: #38ef7d; font-size: 2.5rem; margin: 0; }
.chat-header p { color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; }
.stat-card { background: white; border-radius: 12px; padding: 1.2rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.stat-number { font-size: 2rem; font-weight: 700; color: #302b63; }
.stat-label { font-size: 0.85rem; color: #555; font-weight: 500; }
.source-badge { display: inline-block; background: linear-gradient(90deg, #302b63, #11998e); color: white; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; margin: 4px; }
.stChatMessage { background: white !important; border-radius: 14px !important; box-shadow: 0 2px 12px rgba(0,0,0,0.1) !important; margin-bottom: 1rem !important; }
[data-testid="stChatMessageContent"] p { color: #1a1a2e !important; font-size: 15px !important; line-height: 1.8 !important; }
[data-testid="stChatMessageContent"] li { color: #1a1a2e !important; }
[data-testid="stChatMessageContent"] ol { color: #1a1a2e !important; }
[data-testid="stChatMessageContent"] ul { color: #1a1a2e !important; }
.history-item { background: white; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; border-left: 4px solid #11998e; }
.history-time { font-size: 0.75rem; color: #888; margin-bottom: 4px; }
.history-q { font-size: 0.85rem; font-weight: 600; color: #1a1a2e; }
.history-a { font-size: 0.82rem; color: #444; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
<h1>🏥 Medical RAG Chatbot</h1>
<p>Upload your medical research papers and ask questions using AI</p>
</div>
""", unsafe_allow_html=True)

PDF_FOLDER = "data/raw_pdfs"
CHROMA_DIR = "chroma_db"

@st.cache_resource
def get_models():
    embeddings = get_embedding_model()
    llm = load_llm()
    return embeddings, llm

embeddings, llm = get_models()

def get_or_create_db():
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

with st.sidebar:
    st.markdown("## 📂 Upload Medical PDFs")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("⚙️ Process PDFs", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                os.makedirs(PDF_FOLDER, exist_ok=True)
                saved = 0
                for uploaded_file in uploaded_files:
                    if uploaded_file.size == 0:
                        st.warning(f"Skipped empty file: {uploaded_file.name}")
                        continue
                    file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved += 1
                documents = load_pdfs(PDF_FOLDER)
                if not documents:
                    st.error("No readable text found in PDFs!")
                else:
                    chunks = split_documents(documents)
                    if chunks:
                        db = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=CHROMA_DIR
                        )
                        st.session_state.db = db
                        st.session_state.messages = []
                        pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
                        st.session_state.pdf_count = len(pdfs)
                        st.session_state.chunk_count = len(chunks)
                        st.success(f"Processed {saved} PDF(s) successfully!")
                    else:
                        st.error("Could not extract chunks from PDFs!")

    st.markdown("---")
    st.markdown("### 📄 Current PDFs")
    if os.path.exists(PDF_FOLDER):
        pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
        if pdfs:
            for pdf in pdfs:
                st.markdown(f"- {pdf}")
        else:
            st.markdown("No PDFs loaded yet")

    st.markdown("---")
    if st.button("🗑️ Clear All PDFs", use_container_width=True):
        with st.spinner("Clearing..."):
            if os.path.exists(PDF_FOLDER):
                for f in glob.glob(os.path.join(PDF_FOLDER, "*.pdf")):
                    os.remove(f)
            if "db" in st.session_state:
                try:
                    st.session_state.db._client.reset()
                except:
                    pass
                del st.session_state.db
            st.session_state.pdf_count = 0
            st.session_state.chunk_count = 0
            st.session_state.messages = []
        st.success("Cleared!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 💬 Chat History")
    history = load_history()
    if history:
        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history()
            st.success("History cleared!")
            st.rerun()
        st.markdown(f"**{len([h for h in history if h['role'] == 'user'])} conversations saved**")
        pairs = []
        for i in range(len(history)):
            if history[i]["role"] == "user":
                answer = history[i+1]["content"][:80] + "..." if i+1 < len(history) else ""
                pairs.append((history[i]["timestamp"], history[i]["content"], answer))
        for timestamp, question, answer in pairs[-5:]:
            st.markdown(f"""
            <div class="history-item">
                <div class="history-time">🕐 {timestamp}</div>
                <div class="history-q">Q: {question}</div>
                <div class="history-a">A: {answer}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("No history yet")

if "db" not in st.session_state:
    db = get_or_create_db()
    st.session_state.db = db
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")] if os.path.exists(PDF_FOLDER) else []
    st.session_state.pdf_count = len(pdfs)
    st.session_state.chunk_count = 7

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.get("pdf_count", 0)}</div><div class="stat-label">PDFs Loaded</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.get("chunk_count", 0)}</div><div class="stat-label">Chunks Created</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><div class="stat-number">AI</div><div class="stat-label">Groq Powered</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    saved_history = load_history()
    st.session_state.messages = [{"role": h["role"], "content": h["content"]} for h in saved_history]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("💬 Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    save_message("user", query)
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical documents..."):
            retriever = get_retriever(st.session_state.db)
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""You are a medical assistant. Use the following context from medical research papers to answer the question.

Context:
{context}

Question: {query}

Answer:"""
            response = llm.invoke(prompt)
            answer = response.content

        st.write(answer)
        save_message("assistant", answer)

        sources_html = "<br><b>Sources:</b><br>"
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "?")
            sources_html += f'<span class="source-badge">{source} page {page}</span>'
        st.markdown(sources_html, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})