#!/usr/bin/env python3
"""
Test script to demonstrate the improved Research Assistant
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from packages.agent.agent_autonomous import autonomous_research

def test_improved_research():
    """Test the improved research functionality"""
    
    print("🚀 Testing Improved Research Assistant")
    print("=" * 50)
    
    # Test with a specific research topic
    test_goal = "machine learning applications in healthcare"
    
    print(f"Research Topic: {test_goal}")
    print(f"Expected improvements:")
    print(f"• 40 papers total (20 from arXiv + 20 from OpenAlex)")
    print(f"• Up to 500 documents processed")
    print(f"• 2000 chars per document (vs 1000 before)")
    print(f"• 8000 chars for summarization (vs 3000 before)")
    print(f"• Llama-2-7B model (vs TinyLlama-1.1B before)")
    print(f"• 5+ detailed insights + research questions")
    print()
    
    try:
        # Run with improved settings
        result = autonomous_research(
            goal=test_goal,
            max_results=20,  # 20 from each source = 40 total
            max_docs_to_process=500,
            timeout_minutes=120
        )
        
        if result:
            print("✅ Research completed successfully!")
            print(f"\n📄 Summary Length: {len(result['summary'])} characters")
            print(f"💡 Insights Generated: {len(result['insights'])}")
            
            print(f"\n📄 SUMMARY:")
            print(result['summary'][:500] + "..." if len(result['summary']) > 500 else result['summary'])
            
            print(f"\n💡 INSIGHTS:")
            for i, insight in enumerate(result['insights'][:3], 1):
                print(f"{i}. {insight}")
            
            if len(result['insights']) > 3:
                print(f"... and {len(result['insights']) - 3} more insights")
                
        else:
            print("❌ Research failed to complete")
            
    except Exception as e:
        print(f"❌ Error during research: {e}")
        print("This might be due to missing dependencies or API issues")

if __name__ == "__main__":
    test_improved_research()
