import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize embeddings model - using local HuggingFace model instead of OpenAI to avoid quota issues
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_text_chunks(chunks):
    """
    Generate vector embeddings for a list of text chunks (strings).

    Args:
        chunks (list[str]): List of text segments.

    Returns:
        list[list[float]]: List of embedding vectors.
    """
    return embeddings_model.embed_documents(chunks)

def embed_documents(docs: list[Document]):
    """
    Generate vector embeddings for LangChain Document objects.

    Args:
        docs (list[Document]): List of LangChain documents.

    Returns:
        list[list[float]]: List of embedding vectors.
    """
    # Extract page content from Documents
    contents = [doc.page_content for doc in docs]
    return embeddings_model.embed_documents(contents)

def embed_single_text(text):
    """
    Generate a single vector embedding for a text query.

    Args:
        text (str): Input text.

    Returns:
        list[float]: Embedding vector.
    """
    return embeddings_model.embed_query(text)
