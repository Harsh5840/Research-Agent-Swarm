from packages.retriever.arxiv import search_arxiv
from packages.retriever.openalex import search_openalex
from packages.reader.pdf_reader import extract_text_from_pdf, load_and_split_pdf
from packages.reader.embedder import embed_text_chunks
from packages.rag.vector_store import create_vector_store, similarity_search
from packages.rag.retrieval_chain import build_retrieval_chain, query_retrieval_chain
from packages.summarizer.insight_generator import generate_summary_and_insights
from packages.memory.memory_store import MemoryStore

# Import Document class
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older versions
    from langchain.schema import Document

def autonomous_research(goal: str, max_results: int = 20, persist_path: str = "data/vector_store", max_docs_to_process: int = 500, timeout_minutes: int = 120):
    """
    Autonomous research agent that processes papers and generates insights.
    
    Args:
        goal (str): Research goal/question
        max_results (int): Maximum number of papers to retrieve from each source (increased default: 20)
        persist_path (str): Path to save the vector store
        max_docs_to_process (int): Maximum number of documents to process for vector store (increased default: 500)
        timeout_minutes (int): Timeout in minutes for vector store creation (increased default: 120)
    
    Returns:
        dict: Summary and insights from the research
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

    # Step 2: Process papers into documents
    print(f"[PROCESS] Processing {len(papers)} papers into documents...")
    docs = []
    processed_count = 0
    skipped_count = 0
    
    for i, paper in enumerate(papers):
        try:
            # Show progress every 10 papers
            if i % 10 == 0:
                print(f"[PROCESS] Progress: {i}/{len(papers)} ({i/len(papers)*100:.1f}%)")
            
            # Handle both "url" and "link" fields for compatibility
            paper_url = paper.get("url") or paper.get("link", "")
            source = paper.get("source", "Unknown")
            
            if source == "arXiv":
                # Handle arXiv papers - try PDF first, then fall back to abstract
                pdf_url = paper.get("pdf_url", "")
                abstract = paper.get("summary", "")
                
                if pdf_url:
                    try:
                        text = extract_text_from_pdf(pdf_url)
                        if text and len(text.strip()) > 100:
                            # Split text into smaller chunks to fit LLM context
                            text_chunks = load_and_split_pdf(pdf_url, chunk_size=800, chunk_overlap=100)
                            for i, chunk in enumerate(text_chunks):
                                docs.append(Document(
                                    page_content=chunk.page_content,
                                    metadata={
                                        "title": paper.get("title", "Unknown"),
                                        "url": pdf_url,
                                        "source": source,
                                        "summary": abstract,
                                        "content_type": "full_pdf",
                                        "chunk_index": i,
                                        "total_chunks": len(text_chunks)
                                    }
                                ))
                            processed_count += 1
                            print(f"[PROCESS] Processed arXiv PDF paper {processed_count}: {paper.get('title', 'Unknown')[:50]}... ({len(text_chunks)} chunks)")
                        else:
                            # Fall back to abstract if PDF processing fails
                            if abstract and len(abstract.strip()) > 100:
                                chunks = [abstract[i:i+800] for i in range(0, len(abstract), 700)]
                                for i, chunk in enumerate(chunks):
                                    docs.append(Document(
                                        page_content=chunk,
                                        metadata={
                                            "title": paper.get("title", "Unknown"),
                                            "url": paper_url,
                                            "source": source,
                                            "summary": abstract,
                                            "content_type": "abstract_fallback",
                                            "chunk_index": i,
                                            "total_chunks": len(chunks)
                                        }
                                    ))
                                processed_count += 1
                                print(f"[PROCESS] Processed arXiv abstract paper {processed_count}: {paper.get('title', 'Unknown')[:50]}... ({len(chunks)} chunks)")
                            else:
                                print(f"[WARN] Skipping arXiv paper (insufficient content): {paper.get('title', 'Unknown')}")
                                skipped_count += 1
                    except Exception as e:
                        print(f"[WARN] Failed to process arXiv PDF {paper.get('title', 'Unknown')}: {e}")
                        # Fall back to abstract
                        if abstract and len(abstract.strip()) > 100:
                            chunks = [abstract[i:i+800] for i in range(0, len(abstract), 700)]
                            for i, chunk in enumerate(chunks):
                                docs.append(Document(
                                    page_content=chunk,
                                    metadata={
                                        "title": paper.get("title", "Unknown"),
                                        "url": paper_url,
                                        "source": source,
                                        "summary": abstract,
                                        "content_type": "abstract_fallback",
                                        "chunk_index": i,
                                        "total_chunks": len(chunks)
                                    }
                                ))
                            processed_count += 1
                            print(f"[PROCESS] Processed arXiv abstract paper {processed_count}: {paper.get('title', 'Unknown')[:50]}... ({len(chunks)} chunks)")
                        else:
                            print(f"[WARN] Skipping arXiv paper (insufficient abstract): {paper.get('title', 'Unknown')}")
                            skipped_count += 1
                else:
                    # No PDF URL, use abstract
                    if abstract and len(abstract.strip()) > 100:
                        chunks = [abstract[i:i+800] for i in range(0, len(abstract), 700)]
                        for i, chunk in enumerate(chunks):
                            docs.append(Document(
                                page_content=chunk,
                                metadata={
                                    "title": paper.get("title", "Unknown"),
                                    "url": paper_url,
                                    "source": source,
                                    "summary": abstract,
                                    "content_type": "abstract_only",
                                    "chunk_index": i,
                                    "total_chunks": len(chunks)
                                }
                            ))
                        processed_count += 1
                        print(f"[PROCESS] Processed arXiv abstract paper {processed_count}: {paper.get('title', 'Unknown')[:50]}... ({len(chunks)} chunks)")
                    else:
                        print(f"[WARN] Skipping arXiv paper (insufficient abstract): {paper.get('title', 'Unknown')}")
                        skipped_count += 1
                        
            elif source == "OpenAlex" and paper.get("summary"):
                # Handle OpenAlex papers with abstracts
                abstract = paper.get("summary", "")
                if len(abstract.strip()) > 100:
                    # Split abstract into smaller chunks
                    chunks = [abstract[i:i+800] for i in range(0, len(abstract), 700)]
                    for i, chunk in enumerate(chunks):
                        docs.append(Document(
                            page_content=chunk,
                            metadata={
                                "title": paper.get("title", "Unknown"),
                                "url": paper_url,
                                "source": source,
                                "summary": paper.get("summary", ""),
                                "content_type": "abstract",
                                "authors": paper.get("authors", []),
                                "year": paper.get("year", "Unknown"),
                                "open_access": paper.get("open_access", False),
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                        ))
                    processed_count += 1
                    print(f"[PROCESS] Processed OpenAlex abstract paper {processed_count}: {paper.get('title', 'Unknown')[:50]}... ({len(chunks)} chunks)")
                else:
                    print(f"[WARN] Skipping OpenAlex paper (insufficient abstract): {paper.get('title', 'Unknown')}")
                    skipped_count += 1
            else:
                # Handle other sources
                if paper_url:
                    try:
                        text = extract_text_from_pdf(paper_url)
                        if text and len(text.strip()) > 100:
                            # Split text into smaller chunks to fit LLM context
                            text_chunks = load_and_split_pdf(paper_url, chunk_size=800, chunk_overlap=100)
                            for i, chunk in enumerate(text_chunks):
                                docs.append(Document(
                                    page_content=chunk.page_content,
                                    metadata={
                                        "title": paper.get("title", "Unknown"),
                                        "url": paper_url,
                                        "source": paper.get("source", "Unknown"),
                                        "summary": paper.get("summary", ""),
                                        "content_type": "full_pdf",
                                        "chunk_index": i,
                                        "total_chunks": len(text_chunks)
                                    }
                                ))
                            processed_count += 1
                            print(f"[PROCESS] Processed other paper {processed_count}: {paper.get('title', 'Unknown')[:50]}... ({len(text_chunks)} chunks)")
                        else:
                            print(f"[WARN] Insufficient text extracted from: {paper.get('title', 'Unknown')}")
                            skipped_count += 1
                    except Exception as e:
                        print(f"[WARN] Failed to process {paper.get('title', 'Unknown')}: {e}")
                        skipped_count += 1
                else:
                    print(f"[WARN] No URL for paper: {paper.get('title', 'Unknown')}")
                    skipped_count += 1
                    
            # Check if we've reached the document limit
            if len(docs) >= max_docs_to_process:
                print(f"[PROCESS] Reached document limit of {max_docs_to_process}, stopping processing")
                break
                
        except Exception as e:
            print(f"[ERROR] Unexpected error processing paper {i}: {e}")
            skipped_count += 1
            continue

    if not docs:
        print("[ERROR] No PDFs could be processed successfully.")
        return None

    print(f"[PROCESS] Successfully processed {len(docs)} papers (skipped {skipped_count})")
    print(f"[PROCESS] Final document count: {len(docs)} (limited to {max_docs_to_process})")

    # Step 3: Build vector store
    print("[INDEX] Creating vector store...")
    try:
        # Process more documents for better research coverage
        max_docs_for_vector_store = min(max_docs_to_process, 200)  # Increased from 50 to 200
        if len(docs) > max_docs_for_vector_store:
            print(f"[INDEX] Limiting vector store creation to {max_docs_for_vector_store} documents for optimal processing")
            docs_for_vector_store = docs[:max_docs_for_vector_store]
        else:
            docs_for_vector_store = docs
            
        vector_store = create_vector_store(docs_for_vector_store, persist_path=persist_path, timeout_minutes=timeout_minutes)
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
            
        # Increase content length for better analysis
        max_content_length = 8000  # Increased from 3000 to 8000 for better analysis
        if len(all_text) > max_content_length:
            print(f"[WARN] Content too long ({len(all_text)} chars), truncating to {max_content_length} chars")
            all_text = all_text[:max_content_length] + "... [truncated]"
            
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
