import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pdf_loader import load_pdfs
from chunking import split_documents
from embeddings import get_embedding_model
from rag_pipeline import create_vector_db, load_vector_db, vector_db_exists
from retriever import get_retriever
from chatbot import load_llm

def main():
    embeddings = get_embedding_model()

    if vector_db_exists():
        print("[INFO] Loading existing vector DB...")
        db = load_vector_db(embeddings)
    else:
        print("[INFO] Creating new vector DB...")
        documents = load_pdfs("data/raw_pdfs")
        chunks = split_documents(documents)
        db = create_vector_db(chunks, embeddings)

    retriever = get_retriever(db)
    llm = load_llm()

    print("\n=== Medical RAG Chatbot ===")
    print("Type exit to quit\n")

    while True:
        query = input("Ask a medical question: ")
        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a medical assistant. Use the following context from medical research papers to answer the question.

Context:
{context}

Question: {query}

Answer:"""

        response = llm.invoke(prompt)

        print("\nANSWER:")
        print(response.content)

        print("\nSOURCES:")
        for doc in docs:
            print(f"- {doc.metadata.get('source', 'Unknown')} page {doc.metadata.get('page', '?')}")
        print()

if __name__ == "__main__":
    main()
