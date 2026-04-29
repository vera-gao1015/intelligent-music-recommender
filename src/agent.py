"""
Agentic Workflow module for the Intelligent Music Recommender.

Implements a multi-step reasoning agent that:
1. Analyzes user intent (recommend, explore, compare)
2. Selects an appropriate recommendation strategy
3. Executes the strategy using RAG + scoring
4. Logs all intermediate steps for observability (required by rubric)
"""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# 1. Intent Classification
# ---------------------------------------------------------------------------

INTENT_PROMPT = """You are an intent classifier for a music recommendation system.
Given a user's message, classify it into exactly ONE of these intents:

- "recommend": User wants song recommendations based on mood, activity, or preference.
- "explore": User wants to discover new genres or styles they haven't tried.
- "compare": User wants to compare different types of music or get varied options.

Respond with ONLY a JSON object: {"intent": "<intent>", "reasoning": "<one sentence why>"}
"""


def classify_intent(query: str) -> Dict:
    """
    Step 1: Classify user's intent using LLM.

    Returns:
        Dict with 'intent' and 'reasoning' keys
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()
    # Parse JSON response, with fallback
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"intent": "recommend", "reasoning": "Default fallback due to parsing error"}

    return result


# ---------------------------------------------------------------------------
# 2. Strategy Selection
# ---------------------------------------------------------------------------

STRATEGIES = {
    "recommend": {
        "name": "Mood-Based Recommendation",
        "description": "Retrieve songs matching the user's mood/activity, ranked by relevance.",
        "retrieve_k": 10,
        "recommend_k": 3,
    },
    "explore": {
        "name": "Genre Exploration",
        "description": "Retrieve a diverse set of songs across different genres to broaden taste.",
        "retrieve_k": 15,
        "recommend_k": 5,
    },
    "compare": {
        "name": "Comparative Selection",
        "description": "Retrieve songs from contrasting styles to give the user varied options.",
        "retrieve_k": 15,
        "recommend_k": 4,
    },
}


def select_strategy(intent: str) -> Dict:
    """
    Step 2: Select recommendation strategy based on classified intent.

    Returns:
        Strategy configuration dictionary
    """
    return STRATEGIES.get(intent, STRATEGIES["recommend"])


# ---------------------------------------------------------------------------
# 3. Agent Execution with Observable Steps
# ---------------------------------------------------------------------------

def run_agent(
    query: str,
    songs: List[Dict],
    song_embeddings: List[List[float]],
    rag_retrieve_fn,
    rag_generate_fn,
    verbose: bool = True,
) -> Dict:
    """
    Execute the full agentic workflow with observable intermediate steps.

    Args:
        query: User's natural language input
        songs: Full song catalog
        song_embeddings: Pre-computed embeddings
        rag_retrieve_fn: Function to retrieve songs (from rag.py)
        rag_generate_fn: Function to generate recommendations (from rag.py)
        verbose: If True, print each step to console

    Returns:
        Dict containing all steps and final recommendation
    """
    steps = []

    # ── Step 1: Intent Classification ──────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  AGENT WORKFLOW")
        print("=" * 60)
        print(f"\n[Step 1/4] Classifying user intent...")
        print(f"  Input: \"{query}\"")

    intent_result = classify_intent(query)
    intent = intent_result.get("intent", "recommend")
    reasoning = intent_result.get("reasoning", "")

    step1 = {
        "step": 1,
        "action": "Intent Classification",
        "result": intent,
        "reasoning": reasoning,
    }
    steps.append(step1)

    if verbose:
        print(f"  Intent: {intent}")
        print(f"  Reasoning: {reasoning}")

    # ── Step 2: Strategy Selection ─────────────────────────────────────
    if verbose:
        print(f"\n[Step 2/4] Selecting recommendation strategy...")

    strategy = select_strategy(intent)

    step2 = {
        "step": 2,
        "action": "Strategy Selection",
        "strategy_name": strategy["name"],
        "description": strategy["description"],
        "retrieve_k": strategy["retrieve_k"],
        "recommend_k": strategy["recommend_k"],
    }
    steps.append(step2)

    if verbose:
        print(f"  Strategy: {strategy['name']}")
        print(f"  Description: {strategy['description']}")
        print(f"  Will retrieve {strategy['retrieve_k']} candidates, recommend {strategy['recommend_k']}")

    # ── Step 3: RAG Retrieval ──────────────────────────────────────────
    if verbose:
        print(f"\n[Step 3/4] Retrieving relevant songs via RAG...")

    retrieved = rag_retrieve_fn(
        query, songs, song_embeddings, top_k=strategy["retrieve_k"]
    )

    retrieved_info = [
        {"title": s["title"], "artist": s["artist"], "similarity": round(score, 3)}
        for s, score in retrieved
    ]

    step3 = {
        "step": 3,
        "action": "RAG Retrieval",
        "songs_retrieved": len(retrieved),
        "top_matches": retrieved_info[:5],
    }
    steps.append(step3)

    if verbose:
        print(f"  Retrieved {len(retrieved)} candidates. Top 5:")
        for i, info in enumerate(retrieved_info[:5], 1):
            print(f"    {i}. \"{info['title']}\" by {info['artist']} (similarity: {info['similarity']})")

    # ── Step 4: LLM Generation ─────────────────────────────────────────
    if verbose:
        print(f"\n[Step 4/4] Generating personalized recommendations...")

    recommendation = rag_generate_fn(
        query, retrieved, k=strategy["recommend_k"]
    )

    step4 = {
        "step": 4,
        "action": "LLM Generation",
        "model": "gpt-4o-mini",
        "output_length": len(recommendation),
    }
    steps.append(step4)

    if verbose:
        print(f"  Generated {len(recommendation)} characters of recommendations.")
        print("\n" + "-" * 60)
        print("  RECOMMENDATIONS")
        print("-" * 60)
        print(recommendation)
        print("\n" + "=" * 60)

    return {
        "query": query,
        "intent": intent,
        "strategy": strategy["name"],
        "steps": steps,
        "recommendation": recommendation,
    }
