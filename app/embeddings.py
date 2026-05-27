from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    print("[INFO] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64
        }
    )
    print("[INFO] Embedding model loaded.")
    return embeddings