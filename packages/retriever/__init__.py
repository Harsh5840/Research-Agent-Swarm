from .arxiv import search_arxiv
from .openalex import search_openalex
from .crossref import search_crossref

def search_all_sources(query: str, max_results: int = 5):
    """
    Search all retriever sources (arXiv, OpenAlex, CrossRef) for papers.

    Args:
        query (str): Search term, e.g., "neural architecture search"
        max_results (int): Number of results per source (default=5)

    Returns:
        list[dict]: Combined list of paper metadata sorted by source order.
    """
    results = []

    try:
        results.extend(search_arxiv(query, max_results))
    except Exception as e:
        print(f"[WARN] arXiv fetch failed: {e}")

    try:
        results.extend(search_openalex(query, max_results))
    except Exception as e:
        print(f"[WARN] OpenAlex fetch failed: {e}")

    try:
        results.extend(search_crossref(query, max_results))
    except Exception as e:
        print(f"[WARN] CrossRef fetch failed: {e}")

    return results
