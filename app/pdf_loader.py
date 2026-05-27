from langchain_community.document_loaders import PyPDFLoader
import os


def load_pdfs(pdf_folder: str) -> list:
    documents = []

    if not os.path.exists(pdf_folder):
        print(f"[WARNING] Folder not found: {pdf_folder}")
        return documents

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"[WARNING] No PDFs found in: {pdf_folder}")
        return documents

    for file in pdf_files:
        path = os.path.join(pdf_folder, file)
        print(f"[INFO] Loading: {file}")

        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
            print(f"[INFO] Loaded {len(docs)} pages from {file}")
        except Exception as e:
            print(f"[ERROR] Failed to load {file}: {e}")

    print(f"[INFO] Total pages loaded: {len(documents)}")
    return documents
