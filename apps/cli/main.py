import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent  # Go up one more level to reach research root
sys.path.insert(0, str(project_root))

# Now import the packages
try:
    from packages.agent.agent_autonomous import autonomous_research
    from packages.memory.memory_store import MemoryStore
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Project root: {project_root}")
    print(f"   Python path: {sys.path}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Research Assistant")
    parser.add_argument(
        "goal",
        nargs="?",  # Make goal optional
        type=str,
        help="Research goal or topic to investigate"
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=5,
        help="Max number of papers to retrieve from each source"
    )
    parser.add_argument(
        "--persist_path",
        type=str,
        default="data/vector_store",
        help="Path to save the vector store"
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all previous research sessions"
    )
    parser.add_argument(
        "--show-last",
        action="store_true",
        help="Show the most recent research session"
    )
    args = parser.parse_args()

    # Initialize memory store
    memory = MemoryStore()
    
    # Handle session listing commands
    if args.list_sessions:
        sessions = memory.list_sessions()
        if not sessions:
            print("📚 No previous research sessions found.")
            return
        print(f"📚 Found {len(sessions)} previous research sessions:")
        for i, session in enumerate(sessions, 1):
            print(f"{i}. {session['goal']} - {session['timestamp']}")
        return
    
    if args.show_last:
        last_session = memory.get_last_session()
        if not last_session:
            print("📚 No previous research sessions found.")
            return
        print(f"📚 Last Research Session: {last_session['goal']}")
        print(f"⏰ Date: {last_session['timestamp']}")
        print(f"\n📄 Summary: {last_session['results']['summary']}")
        print(f"\n💡 Insights:")
        for idx, insight in enumerate(last_session['results']['insights'], start=1):
            print(f"{idx}. {insight}")
        return

    # Check if goal is provided for research
    if not args.goal:
        print("❌ Error: Research goal is required for research operations.")
        print("   Use --help to see available options.")
        print("   Use --list-sessions to see previous sessions.")
        print("   Use --show-last to see the most recent session.")
        return

    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found!")
        print("   Please create a .env file with your OPENAI_API_KEY")
        print("   See env.example for reference")
        return

    # Check if OPENAI_API_KEY is set
    from dotenv import load_dotenv
    load_dotenv(env_file)
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("   Please add your OpenAI API key to the .env file")
        return

    print(f"🚀 Starting research on: {args.goal}")
    
    try:
        summary_data = autonomous_research(
            goal=args.goal,
            max_results=args.max_results,
            persist_path=args.persist_path
        )

        if summary_data:
            print("\n📄 SUMMARY:")
            print(summary_data["summary"])
            print("\n💡 INSIGHTS:")
            for idx, insight in enumerate(summary_data["insights"], start=1):
                print(f"{idx}. {insight}")
            
            # Show session storage confirmation
            print(f"\n💾 Research session saved to memory")
            print(f"   Use --list-sessions to see all sessions")
            print(f"   Use --show-last to see the most recent session")
        else:
            print("❌ Research process failed.")
            
    except Exception as e:
        print(f"❌ Error during research: {e}")
        print("   Check your API keys and internet connection")

if __name__ == "__main__":
    main()
