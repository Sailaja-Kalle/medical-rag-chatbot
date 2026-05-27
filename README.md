Here's the complete README file. Copy everything between the lines:

---

```
# 🏥 Medical RAG Chatbot

An AI-powered Medical Chatbot built using **RAG (Retrieval-Augmented Generation)** architecture. Upload any medical research paper and ask questions — the AI retrieves relevant content from your documents and generates accurate answers with source citations.

LIVE DEMO :  Deployed on Streamlit Cloud — [Click here to try it](https://medical-rag-chatbot-project.streamlit.app)

---

## 🧠 What is RAG (Retrieval-Augmented Generation)?

RAG is an AI architecture that combines **document retrieval** with **language model generation**:

- **Without RAG** → LLM answers from training data only (may hallucinate)
- **With RAG** → LLM answers from YOUR documents only (accurate + cited)

### RAG Pipeline Used in This Project

```
INDEXING PIPELINE (runs once when PDF is uploaded)
═══════════════════════════════════════════════════
PDF File
   │
   ▼
[1] PDF Loader ──── Extracts raw text from PDF pages
   │                (LangChain PyPDFLoader)
   ▼
[2] Text Chunker ── Splits text into overlapping chunks
   │                (RecursiveCharacterTextSplitter)
   │                chunk_size=2000, overlap=200
   ▼
[3] Embeddings ──── Converts each chunk to a vector
   │                (HuggingFace all-MiniLM-L6-v2)
   ▼
[4] Vector Store ── Stores all vectors in ChromaDB
                    (Persisted to disk)

QUERY PIPELINE (runs on every question)
═══════════════════════════════════════════════════
User Question
   │
   ▼
[5] Query Embedding ── Converts question to vector
   │
   ▼
[6] Similarity Search ── Finds top 3 similar chunks
   │                     from ChromaDB
   ▼
[7] Context Building ── Combines retrieved chunks
   │
   ▼
[8] Groq LLM ── Generates answer using context
   │            (Llama 3.1 8B Instant)
   ▼
Answer + Source Citations (PDF name + page number)
```

---

## ✨ Features

- 📄 Upload multiple medical PDF research papers from UI
- 🔍 Ask natural language questions about the documents
- 🤖 AI answers generated only from uploaded documents
- 📌 Source citations shown (PDF name + page number)
- 💬 Chat history saved to JSON — persists after closing
- 🗑️ Clear all PDFs and history with one click
- 🎨 Beautiful dark UI with teal/green theme
- ⚡ Batch embeddings for faster CPU processing
- 🔄 ChromaDB persists vectors — no re-processing on restart

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| RAG Framework | LangChain | Orchestrates the full RAG pipeline |
| LLM | Groq API — Llama 3.1 8B | Generates answers from retrieved context |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Converts text to semantic vectors |
| Vector Database | ChromaDB | Stores and retrieves embedding vectors |
| PDF Loader | LangChain PyPDFLoader | Extracts text from PDF pages |
| Text Splitter | RecursiveCharacterTextSplitter | Splits text into overlapping chunks |
| Frontend | Streamlit | Interactive web UI |
| History | JSON file | Persists chat history across sessions |
| Language | Python 3.11 | Core programming language |

---

## 📦 Libraries Used

```
langchain                  # RAG pipeline framework
langchain-community        # Community vector store integrations
langchain-huggingface      # HuggingFace embeddings integration
langchain-groq             # Groq LLM integration
langchain-text-splitters   # Text chunking utilities
chromadb                   # Local vector database
sentence-transformers      # HuggingFace embedding model
streamlit                  # Web UI framework
pypdf                      # PDF text extraction
python-dotenv              # Load API keys from .env file
groq                       # Groq API client
```

---

## 🗂️ Project Structure

```
medical-rag-chatbot/
│
├── app/
│   ├── main.py            # Terminal chatbot entry point
│   ├── rag_pipeline.py    # RAG — Vector DB creation and loading
│   ├── pdf_loader.py      # RAG — PDF loading using PyPDFLoader
│   ├── chunking.py        # RAG — Text splitting into chunks
│   ├── embeddings.py      # RAG — HuggingFace embedding model
│   ├── retriever.py       # RAG — ChromaDB similarity retriever
│   ├── chatbot.py         # RAG — Groq LLM loader
│   ├── prompts.py         # RAG — Prompt templates
│   └── utils.py           # Chat history save/load utilities
│
├── data/
│   ├── raw_pdfs/          # Uploaded PDF files (gitignored)
│   └── chat_history.json  # Saved chat history (gitignored)
│
├── chroma_db/             # ChromaDB vector storage (gitignored)
│
├── frontend/
│   └── streamlit_app.py   # Main Streamlit UI with RAG integration
│
├── requirements.txt       # Python dependencies
├── .env                   # API keys (gitignored)
└── README.md              # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sailaja-Kalle/medical-rag-chatbot.git
cd medical-rag-chatbot
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run frontend/streamlit_app.py
```

Open browser at: http://localhost:8501

---

## 💡 How to Use

1. Open the app at `http://localhost:8501`
2. Upload medical PDF files from the left sidebar
3. Click **Process PDFs** button
4. Type your question in the chat box
5. Get AI answer with source citations!

---

## 🔑 API Keys Required

| API | Free? | Get it here |
|---|---|---|
| Groq API | ✅ Free | https://console.groq.com |
| HuggingFace | ✅ Free (runs locally) | https://huggingface.co |

---

## 📊 Performance Notes

| Scenario | Speed |
|---|---|
| Small PDF (1-10 pages) | ~5-10 seconds |
| Medium PDF (10-50 pages) | ~30-60 seconds |
| Large PDF (50-100 pages) | ~1-2 minutes |
| Query response time | ~2-5 seconds |

- Batch size 64 enabled for faster CPU embedding
- ChromaDB persists vectors — reload is instant after first processing
- No GPU required — runs on CPU

---

## 🙋 Author

**Sailaja Kalle**
- GitHub: https://github.com/Sailaja-Kalle

---

