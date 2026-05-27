def get_retriever(db):
    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )
    print("[INFO] Retriever ready.")
    return retriever
