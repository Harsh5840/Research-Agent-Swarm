import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from dotenv import load_dotenv
import os

# Load .env variables if needed
load_dotenv()

# Default to official arXiv API if not in .env
ARXIV_API_URL = os.getenv("ARXIV_API_URL", "https://export.arxiv.org/api/query")

def search_arxiv(query: str, max_results: int = 5):
    """
    Search arXiv for research papers matching the query.
    
    Args:
        query (str): Search term, e.g., "neural architecture search"
        max_results (int): Number of results to fetch (default=5)

    Returns:
        list[dict]: List of paper metadata {title, summary, link, source}
    """
    # URL encode the search query
    url = f"{ARXIV_API_URL}?search_query={quote(query)}&start=0&max_results={max_results}"
    
    # Send HTTP GET request
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # Parse XML response (Atom format)
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []

    for entry in root.findall("atom:entry", ns):
        title_elem = entry.find("atom:title", ns)
        summary_elem = entry.find("atom:summary", ns)
        link_elem = entry.find("atom:id", ns)

        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        summary = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
        link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""

        results.append({
            "title": title,
            "summary": summary,
            "link": link,
            "source": "arXiv"
        })

    return results
