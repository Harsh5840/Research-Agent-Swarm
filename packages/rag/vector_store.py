import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Ensure API key exists
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# Initialize embeddings model - using local HuggingFace model instead of OpenAI to avoid quota issues
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(docs: list[Document], persist_path: str = None):
    """
    Create a FAISS vector store from LangChain Document objects.

    Args:
        docs (list[Document]): List of LangChain documents.
        persist_path (str): Optional path to save the FAISS index.

    Returns:
        FAISS: The FAISS vector store object.
    """
    vector_store = FAISS.from_documents(docs, embeddings_model)
    if persist_path:
        vector_store.save_local(persist_path)
    return vector_store

def load_vector_store(persist_path: str):
    """
    Load a FAISS vector store from disk.

    Args:
        persist_path (str): Path where the FAISS index was saved.

    Returns:
        FAISS: Loaded FAISS vector store object.
    """
    return FAISS.load_local(persist_path, embeddings_model, allow_dangerous_deserialization=True)

def similarity_search(vector_store: FAISS, query: str, k: int = 5):
    """
    Perform a similarity search in the vector store.

    Args:
        vector_store (FAISS): The vector store object.
        query (str): Search query.
        k (int): Number of results to return.

    Returns:
        list[Document]: List of most similar documents.
    """
    return vector_store.similarity_search(query, k=k)
