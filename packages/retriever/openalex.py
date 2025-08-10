import requests
from urllib.parse import quote
from dotenv import load_dotenv
import os

# Load .env variables if any
load_dotenv()

# Default to official OpenAlex API if not set
OPENALEX_API_URL = os.getenv("OPENALEX_API_URL", "https://api.openalex.org")

def search_openalex(query: str, per_page: int = 5):
    """
    Search OpenAlex for research papers matching the query.

    Args:
        query (str): Search term, e.g., "neural architecture search"
        per_page (int): Number of results to fetch (default=5)

    Returns:
        list[dict]: List of paper metadata {title, summary, link, source}
    """
    # URL encode the search query
    url = f"{OPENALEX_API_URL}/works?search={quote(query)}&per-page={per_page}"

    # Send HTTP GET request
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # Parse JSON response
    data = response.json()
    results = []

    for work in data.get("results", []):
        abstract_dict = work.get("abstract_inverted_index", None)

        # Convert abstract_inverted_index to a string if available
        if abstract_dict:
            abstract_words = sorted(
                [(pos, word) for word, positions in abstract_dict.items() for pos in positions],
                key=lambda x: x[0]
            )
            abstract_text = " ".join(word for _, word in abstract_words)
        else:
            abstract_text = ""

        results.append({
            "title": work.get("display_name", "").strip(),
            "summary": abstract_text.strip(),
            "link": work.get("id", "").strip(),
            "source": "OpenAlex"
        })

    return results
