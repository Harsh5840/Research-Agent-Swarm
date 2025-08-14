import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever

load_dotenv()

def build_retrieval_chain(vector_store: FAISS):
    """
    Build a LangChain RetrievalQA chain using a vector store.

    Args:
        vector_store (FAISS): Loaded or freshly created vector store.

    Returns:
        RetrievalQA: Configured RAG chain.
    """
    # Use built-in retriever with limited search results
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )

    # Use local LLM instead of OpenAI to avoid API quota issues
    llm = CTransformers(
        model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        model_type="llama",
        config={'max_new_tokens': 256, 'temperature': 0.1, 'context_length': 2048}
    )

    # Create a custom prompt template that limits context length
    from langchain.prompts import PromptTemplate
    
    # Limit the context to prevent overflow
    qa_prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""

    qa_prompt = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
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
    import threading
    import time
    
    result = None
    error = None
    
    def run_query():
        nonlocal result, error
        try:
            result = chain.invoke({"query": query})
        except Exception as e:
            error = e
    
    # Run query in a separate thread with timeout
    thread = threading.Thread(target=run_query)
    thread.daemon = True
    thread.start()
    thread.join(timeout=60)  # 60 second timeout
    
    if thread.is_alive():
        print("[WARN] Retrieval chain query timed out, returning empty result")
        return {"result": "Query timed out", "source_documents": []}
    elif error:
        print(f"[WARN] Retrieval chain query failed: {error}")
        return {"result": f"Query failed: {error}", "source_documents": []}
    else:
        return result
