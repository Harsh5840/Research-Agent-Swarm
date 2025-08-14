"""
Configuration settings for the Research Assistant
"""

# Search Configuration
DEFAULT_MAX_RESULTS = 20  # Papers per source (arXiv + OpenAlex = 40 total)
DEFAULT_MAX_DOCS_PROCESS = 500  # Maximum documents to process
DEFAULT_TIMEOUT_MINUTES = 120  # Timeout for vector store creation

# Content Processing
MAX_DOCUMENT_CHARS = 2000  # Characters per document for embeddings
MAX_SUMMARY_CHARS = 8000  # Characters for summarization
VECTOR_STORE_BATCH_SIZE = 50  # Documents per batch

# Model Configuration
PREFERRED_MODEL = "openai"  # "openai" or "local"
LOCAL_MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGUF"
OPENAI_MODEL = "gpt-3.5-turbo"

# Quality Settings
ENABLE_DETAILED_ANALYSIS = True  # Generate more detailed insights
MIN_INSIGHTS_COUNT = 5  # Minimum number of insights to generate
INCLUDE_RESEARCH_QUESTIONS = True  # Generate future research questions

# Performance Settings
ENABLE_CHECKPOINTS = True  # Save progress during processing
ENABLE_MEMORY_PERSISTENCE = True  # Save research sessions
