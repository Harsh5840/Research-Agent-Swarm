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
        list[dict]: List of paper metadata {title, summary, link, source, pdf_url, authors, year}
    """
    # URL encode the search query
    url = f"{OPENALEX_API_URL}/works?search={quote(query)}&per-page={per_page}&select=id,display_name,abstract_inverted_index,publication_year,authorships,open_access,locations"

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

        # Try to find PDF URL from locations
        pdf_url = None
        locations = work.get("locations", [])
        for location in locations:
            if location.get("type") == "pdf":
                pdf_url = location.get("pdf_url")
                break

        # Get authors
        authors = []
        authorships = work.get("authorships", [])
        for authorship in authorships:
            author = authorship.get("author", {})
            if author.get("display_name"):
                authors.append(author.get("display_name"))

        # Get publication year
        year = work.get("publication_year", "Unknown")

        results.append({
            "title": work.get("display_name", "").strip(),
            "summary": abstract_text.strip(),
            "link": work.get("id", "").strip(),
            "pdf_url": pdf_url,
            "source": "OpenAlex",
            "authors": authors,
            "year": year,
            "open_access": work.get("open_access", {}).get("is_oa", False)
        })

    return results
