import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    print("[INFO] Loading Groq LLM...")
    api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=api_key
    )
    print("[INFO] LLM loaded.")
    return llm