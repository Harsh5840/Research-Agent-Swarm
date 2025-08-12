import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# Prompt template for summarization & insight generation
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["content"],
    template="""
You are an AI research assistant. Given the extracted research content below,
summarize the key findings, methods, and conclusions in a concise and clear way.

Then, generate 3 unique insights or next-step research ideas based on the text.

Content:
{content}

---
Summary:
Insights:
"""
)

def generate_summary_and_insights(content: str) -> dict:
    """
    Summarize research content and generate insights.

    Args:
        content (str): Extracted research text.

    Returns:
        dict: { 'summary': str, 'insights': list[str] }
    """
    prompt = SUMMARY_PROMPT.format(content=content)
    response = llm.invoke(prompt).content

    # Simple parsing: Split summary and insights based on section keywords
    parts = response.split("Insights:")
    summary = parts[0].replace("Summary:", "").strip()
    insights = parts[1].strip().split("\n") if len(parts) > 1 else []

    # Clean empty lines
    insights = [i.strip("-• ").strip() for i in insights if i.strip()]

    return {
        "summary": summary,
        "insights": insights
    }
