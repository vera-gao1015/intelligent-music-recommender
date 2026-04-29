"""
Intelligent Music Recommender - Main Entry Point

An AI-powered music recommendation system that uses:
- RAG (Retrieval-Augmented Generation) for semantic song search
- Agentic workflow for multi-step reasoning
- Guardrails for reliability and safety

Usage:
    python3 -m src.main
"""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import load_songs
from src.rag import build_song_embeddings, retrieve_songs, generate_recommendation
from src.agent import run_agent
from src.guardrails import validate_input, run_guardrails, logger


def main():
    """Run the interactive music recommender."""

    print("\n" + "=" * 60)
    print("  Intelligent Music Recommender")
    print("  Powered by RAG + Agentic Workflow + GPT-4o-mini")
    print("=" * 60)

    # Load songs and build embeddings
    print("\n[System] Loading song catalog...")
    songs = load_songs("data/songs.csv")
    print(f"[System] Loaded {len(songs)} songs.")

    print("[System] Building embeddings...")
    embeddings = build_song_embeddings(songs)
    print("[System] Ready!\n")

    # Interactive loop
    while True:
        print("-" * 60)
        query = input("What kind of music are you looking for? (type 'quit/exit/q' to exit)\n> ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("\nThanks for using Intelligent Music Recommender! Goodbye!")
            break

        # Guardrail: validate input
        is_valid, msg = validate_input(query)
        if not is_valid:
            print(f"\n[Guardrail] {msg}")
            continue

        # Run the agent workflow
        result = run_agent(
            query=query,
            songs=songs,
            song_embeddings=embeddings,
            rag_retrieve_fn=retrieve_songs,
            rag_generate_fn=generate_recommendation,
            verbose=True,
        )

        # Guardrail: verify output
        retrieved = retrieve_songs(query, songs, embeddings, top_k=10)
        report = run_guardrails(
            query=query,
            recommendation=result["recommendation"],
            retrieved_songs=retrieved,
            expected_count=3,
        )

        # Print guardrail report
        print("\n[Guardrail Report]")
        print(f"  Input Validation:    {'PASS' if report['input_validation']['passed'] else 'FAIL'}")
        print(f"  Output Verification: {'PASS' if report['output_verification']['passed'] else 'FAIL'}")
        print(f"  Quality Check:       {'PASS' if report['quality_check']['passed'] else 'FAIL'}")

        if report["output_verification"]["warnings"]:
            for w in report["output_verification"]["warnings"]:
                print(f"  Warning: {w}")
        if report["quality_check"]["issues"]:
            for issue in report["quality_check"]["issues"]:
                print(f"  Issue: {issue}")

        print()


if __name__ == "__main__":
    main()
