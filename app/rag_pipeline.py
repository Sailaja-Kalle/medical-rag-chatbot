from langchain_community.vectorstores import Chroma
import os

CHROMA_DIR = "chroma_db"

def create_vector_db(chunks, embeddings):
    print(f"[INFO] Creating vector database...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"[INFO] Vector DB saved!")
    return db

def load_vector_db(embeddings):
    print(f"[INFO] Loading existing vector DB...")
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return db

def vector_db_exists():
    return os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0
