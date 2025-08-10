import requests
from urllib.parse import quote
from dotenv import load_dotenv
import os

# Load .env variables if any
load_dotenv()

# Default to official CrossRef API if not set
CROSSREF_API_URL = os.getenv("CROSSREF_API_URL", "https://api.crossref.org")

def search_crossref(query: str, rows: int = 5):
    """
    Search CrossRef for research papers matching the query.

    Args:
        query (str): Search term, e.g., "neural architecture search"
        rows (int): Number of results to fetch (default=5)

    Returns:
        list[dict]: List of paper metadata {title, summary, link, source}
    """
    # URL encode the search query
    url = f"{CROSSREF_API_URL}/works?query={quote(query)}&rows={rows}"

    # Send HTTP GET request
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # Parse JSON response
    data = response.json()
    results = []

    for item in data.get("message", {}).get("items", []):
        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""

        abstract = item.get("abstract", "")
        # CrossRef abstracts are often in HTML-ish tags, strip them crudely
        abstract = abstract.replace("<jats:p>", "").replace("</jats:p>", "").strip()

        link = ""
        if "DOI" in item:
            link = f"https://doi.org/{item['DOI']}"

        results.append({
            "title": title.strip(),
            "summary": abstract.strip(),
            "link": link.strip(),
            "source": "CrossRef"
        })

    return results
