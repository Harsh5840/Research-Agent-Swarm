import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Full text content from the PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()

    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Splits text into overlapping chunks for embedding using LangChain's
    RecursiveCharacterTextSplitter.

    Args:
        text (str): The text to chunk.
        chunk_size (int): Number of characters per chunk.
        overlap (int): Overlap between chunks.

    Returns:
        list[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return splitter.split_text(text)
