from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    print("[INFO] Loading Groq LLM...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )
    print("[INFO] LLM loaded.")
    return llm
