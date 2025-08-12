# packages/reader/pdf_reader.py

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path: str, chunk_size=1000, chunk_overlap=200):
    """Loads a PDF and splits it into LangChain Documents."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(pages)

def extract_text_from_pdf(file_path: str) -> str:
    """Loads a PDF and returns its full text as one string."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return "\n\n".join([page.page_content for page in pages])
