import argparse
from packages.agent.agent_autonomous import autonomous_research

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Research Assistant")
    parser.add_argument(
        "goal",
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
    args = parser.parse_args()

    print(f"🚀 Starting research on: {args.goal}")
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
    else:
        print("❌ Research process failed.")

if __name__ == "__main__":
    main()
