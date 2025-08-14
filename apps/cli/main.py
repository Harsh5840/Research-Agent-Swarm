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
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🚀 Autonomous Research Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic research with default settings
  python main.py "What are the latest advances in transformer models?"
  
  # Research with custom parameters
  python main.py "Machine learning applications in healthcare" --max_results 10 --max_docs 100 --timeout 30
  
  # Clean up checkpoint files from failed runs
  python main.py --cleanup-checkpoints
  
  # List previous research sessions
  python main.py --list-sessions
  
  # Show the most recent session
  python main.py --show-last

Tips:
  • Use --max_docs to limit processing for faster results
  • Use --timeout to prevent hanging on large datasets
  • Use --cleanup-checkpoints if you encounter errors
  • The system automatically saves progress and can resume from failures
        """
    )
    parser.add_argument(
        "goal",
        nargs="?",  # Make goal optional
        type=str,
        help="Research goal or topic to investigate"
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=20,  # Increased from 5 to 20
        help="Max number of papers to retrieve from each source"
    )
    parser.add_argument(
        "--persist_path",
        type=str,
        default="data/vector_store",
        help="Path to save the vector store"
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=500,  # Increased from 200 to 500
        help="Maximum number of documents to process for vector store (default: 500)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,  # Increased from 60 to 120
        help="Timeout in minutes for vector store creation (default: 120)"
    )
    parser.add_argument(
        "--cleanup-checkpoints",
        action="store_true",
        help="Clean up checkpoint files from failed runs"
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
    
    if args.cleanup_checkpoints:
        from packages.rag.vector_store import cleanup_checkpoints
        cleanup_checkpoints(args.persist_path)
        print("✅ Checkpoint cleanup completed")
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
        print("   Note: OpenAI API key is no longer required for local operation")
        print("   See env.example for reference if you want to use OpenAI features")
        return

    # Load environment variables (optional now)
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    print(f"🚀 Starting research on: {args.goal}")
    
    try:
        summary_data = autonomous_research(
            goal=args.goal,
            max_results=args.max_results,
            persist_path=args.persist_path,
            max_docs_to_process=args.max_docs,
            timeout_minutes=args.timeout
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
