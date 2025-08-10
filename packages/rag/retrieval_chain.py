import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

def build_retrieval_chain(vector_store: FAISS):
    """
    Build a LangChain RetrievalQA chain using a vector store.

    Args:
        vector_store (FAISS): Loaded or freshly created vector store.

    Returns:
        RetrievalQA: Configured RAG chain.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def query_retrieval_chain(chain: RetrievalQA, query: str):
    """
    Query the retrieval QA chain.

    Args:
        chain (RetrievalQA): The RAG chain instance.
        query (str): User's question.

    Returns:
        dict: Answer and source documents.
    """
    return chain.invoke({"query": query})
