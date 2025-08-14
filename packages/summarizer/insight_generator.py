import os
from dotenv import load_dotenv
from langchain_community.llms import CTransformers
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

# Check if OpenAI API key is available
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM - prefer OpenAI for better quality, fallback to local model
if openai_api_key:
    print("[SUMMARIZER] Using OpenAI for high-quality analysis")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=1000
    )
else:
    print("[SUMMARIZER] Using local Llama-2-7B model (OpenAI not available)")
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        model_type="llama",
        config={'max_new_tokens': 512, 'temperature': 0.3, 'context_length': 4096}
    )

# Prompt template for summarization & insight generation
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["content"],
    template="""
You are an expert AI research assistant with deep knowledge in academic research. Given the extracted research content below, provide a comprehensive analysis.

Please provide:

1. A detailed summary of the key findings, methodologies, and conclusions from the research papers
2. 5 unique insights that highlight important trends, gaps, or opportunities for future research
3. 3 specific research questions that could be explored further based on this content

Focus on extracting the most valuable and actionable information from the research.

Content:
{content}

---
Detailed Summary:

Key Insights:
1. 
2. 
3. 
4. 
5. 

Future Research Questions:
1. 
2. 
3. 
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
    response = llm.invoke(prompt)
    
    # Handle different response formats from OpenAI vs local models
    if hasattr(response, 'content'):
        response_text = response.content
    elif hasattr(response, 'text'):
        response_text = response.text
    else:
        response_text = str(response)

    # Enhanced parsing: Split summary and insights based on section keywords
    parts = response_text.split("Key Insights:")
    summary = parts[0].replace("Detailed Summary:", "").strip()
    
    if len(parts) > 1:
        insights_part = parts[1]
        # Split insights and research questions
        if "Future Research Questions:" in insights_part:
            insights_section, questions_section = insights_part.split("Future Research Questions:")
            insights = [i.strip("-• ").strip() for i in insights_section.strip().split("\n") if i.strip() and not i.strip().startswith("Future")]
            questions = [q.strip("-• ").strip() for q in questions_section.strip().split("\n") if q.strip()]
        else:
            insights = [i.strip("-• ").strip() for i in insights_part.strip().split("\n") if i.strip()]
            questions = []
    else:
        insights = []
        questions = []

    # Clean empty lines and combine insights with questions
    insights = [i for i in insights if i and not i.isdigit()]
    questions = [q for q in questions if q and not q.isdigit()]
    
    # Combine insights and questions for backward compatibility
    all_insights = insights + questions

    return {
        "summary": summary,
        "insights": all_insights
    }
