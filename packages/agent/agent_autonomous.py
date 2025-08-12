from packages.retriever.arxiv import search_arxiv
from packages.retriever.openalex import search_openalex
from packages.reader.pdf_reader import extract_text_from_pdf
from packages.reader.embedder import embed_text_chunks
from packages.rag.vector_store import create_vector_store, similarity_search
from packages.rag.retrieval_chain import build_retrieval_chain, query_retrieval_chain
from packages.summarizer.insight_generator import generate_summary_and_insights
from packages.memory.memory_store import MemoryStore

def autonomous_research(goal: str, max_results: int = 5, persist_path: str = "data/vector_store"):
    """
    Run a full autonomous research loop:
    1. Search papers
    2. Read & embed them
    3. Store in vector DB
    4. Query for insights
    5. Save to memory
    """
    memory = MemoryStore()

    # Step 1: Search sources
    print(f"[SEARCH] Looking for papers on '{goal}'...")
    arxiv_results = search_arxiv(goal, max_results=max_results)
    openalex_results = search_openalex(goal, max_results=max_results)

    papers = arxiv_results + openalex_results
    if not papers:
        print("[ERROR] No papers found.")
        return None

    # Step 2: Extract text & create LangChain documents
    from langchain.schema import Document
    docs = []
    for paper in papers:
        if not paper.get("pdf_url"):
            continue
        text = extract_text_from_pdf(paper["pdf_url"])
        if text:
            docs.append(Document(page_content=text, metadata={"title": paper["title"], "url": paper["pdf_url"]}))

    if not docs:
        print("[ERROR] No PDFs could be processed.")
        return None

    # Step 3: Build vector store
    print("[INDEX] Creating vector store...")
    vector_store = create_vector_store(docs, persist_path=persist_path)

    # Step 4: Build retrieval chain & query
    retrieval_chain = build_retrieval_chain(vector_store)
    rag_response = query_retrieval_chain(retrieval_chain, goal)

    # Step 5: Generate summary & insights
    all_text = "\n\n".join([doc.page_content for doc in rag_response["source_documents"]])
    summary_data = generate_summary_and_insights(all_text)

    # Step 6: Save to memory
    print("[SAVE] Storing session in memory...")
    memory.add_session(goal, {
        "summary": summary_data["summary"],
        "insights": summary_data["insights"],
        "sources": [doc.metadata for doc in rag_response["source_documents"]]
    })

    return summary_data
