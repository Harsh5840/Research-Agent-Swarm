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
    try:
        arxiv_results = search_arxiv(goal, max_results=max_results)
        print(f"[SEARCH] Found {len(arxiv_results)} papers from arXiv")
    except Exception as e:
        print(f"[WARN] arXiv search failed: {e}")
        arxiv_results = []
    
    try:
        openalex_results = search_openalex(goal, per_page=max_results)
        print(f"[SEARCH] Found {len(openalex_results)} papers from OpenAlex")
    except Exception as e:
        print(f"[WARN] OpenAlex search failed: {e}")
        openalex_results = []

    papers = arxiv_results + openalex_results
    if not papers:
        print("[ERROR] No papers found from any source.")
        return None

    print(f"[SEARCH] Total papers found: {len(papers)}")

    # Step 2: Extract text & create LangChain documents
    try:
        from langchain_core.documents import Document
    except ImportError:
        # Fallback for older versions
        from langchain.schema import Document
    
    docs = []
    processed_count = 0
    
    for paper in papers:
        # Handle both "link" and "pdf_url" fields
        paper_url = paper.get("pdf_url") or paper.get("link")
        if not paper_url:
            print(f"[WARN] No URL for paper: {paper.get('title', 'Unknown')}")
            continue
            
        # Check if this is a PDF URL or if we need to convert it
        if paper.get("source") == "arXiv":
            # arXiv links need to be converted to PDF URLs
            if "arxiv.org/abs/" in paper_url:
                pdf_url = paper_url.replace("/abs/", "/pdf/") + ".pdf"
            else:
                pdf_url = paper_url
        elif paper.get("source") == "OpenAlex":
            # OpenAlex doesn't provide direct PDF URLs, skip for now
            print(f"[WARN] Skipping OpenAlex paper (no PDF): {paper.get('title', 'Unknown')}")
            continue
        else:
            pdf_url = paper_url
            
        try:
            text = extract_text_from_pdf(pdf_url)
            if text and len(text.strip()) > 100:  # Ensure we have meaningful content
                docs.append(Document(
                    page_content=text, 
                    metadata={
                        "title": paper.get("title", "Unknown"),
                        "url": pdf_url,
                        "source": paper.get("source", "Unknown"),
                        "summary": paper.get("summary", "")
                    }
                ))
                processed_count += 1
                print(f"[PROCESS] Processed paper {processed_count}: {paper.get('title', 'Unknown')[:50]}...")
            else:
                print(f"[WARN] Insufficient text extracted from: {paper.get('title', 'Unknown')}")
        except Exception as e:
            print(f"[WARN] Failed to process {paper.get('title', 'Unknown')}: {e}")

    if not docs:
        print("[ERROR] No PDFs could be processed successfully.")
        return None

    print(f"[PROCESS] Successfully processed {len(docs)} papers")

    # Step 3: Build vector store
    print("[INDEX] Creating vector store...")
    try:
        vector_store = create_vector_store(docs, persist_path=persist_path)
        print("[INDEX] Vector store created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create vector store: {e}")
        return None

    # Step 4: Build retrieval chain & query
    try:
        retrieval_chain = build_retrieval_chain(vector_store)
        print("[QUERY] Querying retrieval chain...")
        rag_response = query_retrieval_chain(retrieval_chain, goal)
        print("[QUERY] Retrieval chain query completed")
    except Exception as e:
        print(f"[ERROR] Failed to query retrieval chain: {e}")
        return None

    # Step 5: Generate summary & insights
    try:
        all_text = "\n\n".join([doc.page_content for doc in rag_response.get("source_documents", [])])
        if not all_text.strip():
            print("[WARN] No source documents found for summarization")
            return None
            
        summary_data = generate_summary_and_insights(all_text)
        print("[SUMMARY] Summary and insights generated successfully")
    except Exception as e:
        print(f"[ERROR] Failed to generate summary: {e}")
        return None

    # Step 6: Save to memory
    print("[SAVE] Storing session in memory...")
    try:
        memory.add_session(goal, {
            "summary": summary_data["summary"],
            "insights": summary_data["insights"],
            "sources": [doc.metadata for doc in rag_response.get("source_documents", [])],
            "total_papers_found": len(papers),
            "successfully_processed": len(docs)
        })
        print("[SAVE] Session saved to memory successfully")
    except Exception as e:
        print(f"[WARN] Failed to save session to memory: {e}")

    return summary_data
