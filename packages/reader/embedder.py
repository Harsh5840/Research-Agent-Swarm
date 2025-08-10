import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Ensure API key is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

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
