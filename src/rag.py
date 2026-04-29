"""
RAG (Retrieval-Augmented Generation) module for the Intelligent Music Recommender.

This module:
1. Converts song metadata into text descriptions
2. Generates embeddings using OpenAI's embedding model
3. Retrieves the most relevant songs based on user's natural language query
4. Uses GPT to generate personalized recommendation explanations
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# 1. Song Text Representation
# ---------------------------------------------------------------------------

def song_to_text(song: Dict) -> str:
    """
    Convert a song's metadata into a natural language description.
    This text is used to generate embeddings for semantic search.
    """
    return (
        f'"{song["title"]}" by {song["artist"]} is a {song["genre"]} song '
        f'with a {song["mood"]} mood. '
        f'Energy level: {song["energy"]:.2f}, '
        f'tempo: {song["tempo_bpm"]} BPM, '
        f'valence: {song["valence"]:.2f}, '
        f'danceability: {song["danceability"]:.2f}, '
        f'acousticness: {song["acousticness"]:.2f}.'
    )


# ---------------------------------------------------------------------------
# 2. Embedding Generation & Caching
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings_cache.json")


def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a single text string using OpenAI API."""
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def build_song_embeddings(songs: List[Dict], force_rebuild: bool = False) -> List[List[float]]:
    """
    Build embeddings for all songs. Uses a local JSON cache to avoid
    redundant API calls (saves cost and speeds up subsequent runs).

    Args:
        songs: List of song dictionaries
        force_rebuild: If True, ignore cache and rebuild all embeddings

    Returns:
        List of embedding vectors, one per song
    """
    # Try loading from cache
    if not force_rebuild and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            cache = json.load(f)
        # Validate cache matches current song count
        if len(cache.get("embeddings", [])) == len(songs):
            print("[RAG] Loaded embeddings from cache.")
            return cache["embeddings"]

    # Generate embeddings via API
    print(f"[RAG] Generating embeddings for {len(songs)} songs...")
    descriptions = [song_to_text(song) for song in songs]

    # Batch API call (more efficient than one-by-one)
    response = client.embeddings.create(
        input=descriptions,
        model=EMBEDDING_MODEL,
    )
    embeddings = [item.embedding for item in response.data]

    # Save to cache
    cache = {
        "model": EMBEDDING_MODEL,
        "song_count": len(songs),
        "embeddings": embeddings,
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print("[RAG] Embeddings cached to disk.")

    return embeddings


# ---------------------------------------------------------------------------
# 3. Semantic Retrieval
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def retrieve_songs(
    query: str,
    songs: List[Dict],
    song_embeddings: List[List[float]],
    top_k: int = 10,
) -> List[Tuple[Dict, float]]:
    """
    Retrieve the most relevant songs for a natural language query.

    Args:
        query: User's natural language description (e.g., "I'm tired and want to relax")
        songs: Full list of song dictionaries
        song_embeddings: Pre-computed embedding vectors for each song
        top_k: Number of songs to retrieve

    Returns:
        List of (song_dict, similarity_score) tuples, sorted by relevance
    """
    query_embedding = get_embedding(query)

    scored = []
    for song, emb in zip(songs, song_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scored.append((song, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# 4. LLM-Powered Recommendation Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly and knowledgeable music recommendation assistant.
Based on the user's mood or request and the retrieved song data, provide personalized
music recommendations with warm, engaging explanations.

Rules:
- Recommend exactly {k} songs from the provided candidates.
- For each song, explain WHY it fits the user's current mood or request.
- Keep each explanation concise (1-2 sentences).
- Reference specific song attributes (genre, mood, energy, tempo) in your reasoning.
- Use a conversational, friendly tone.
- Format your response as a numbered list.
- If the user's request is vague, make reasonable inferences about what they might enjoy.
"""

FEW_SHOT_USER = """User request: "I need something chill to study to"

Retrieved songs:
1. "Midnight Coding" by LoRoom - lofi, chill, energy: 0.42, tempo: 78 BPM, acousticness: 0.71
2. "Focus Flow" by LoRoom - lofi, focused, energy: 0.40, tempo: 80 BPM, acousticness: 0.78
3. "Library Rain" by Paper Lanterns - lofi, chill, energy: 0.35, tempo: 72 BPM, acousticness: 0.86

Recommend top 3 songs."""

FEW_SHOT_ASSISTANT = """Here are my top 3 picks for your study session:

1. **"Focus Flow" by LoRoom** - This lofi track is literally built for concentration. With a focused mood, low energy (0.40), and a steady 80 BPM tempo, it creates the perfect background for deep work without being distracting.

2. **"Library Rain" by Paper Lanterns** - The high acousticness (0.86) and gentle 72 BPM tempo make this feel like a cozy library session. Its chill vibe will keep you relaxed while you power through your notes.

3. **"Midnight Coding" by LoRoom** - Another great lofi option with a chill mood and moderate acousticness (0.71). The slightly higher energy (0.42) compared to the others gives you just enough momentum to stay productive."""


def generate_recommendation(
    query: str,
    retrieved_songs: List[Tuple[Dict, float]],
    k: int = 3,
) -> str:
    """
    Use GPT to generate personalized recommendations based on retrieved songs.

    Args:
        query: User's original natural language request
        retrieved_songs: List of (song_dict, similarity_score) from retrieval step
        k: Number of songs to recommend in final output

    Returns:
        LLM-generated recommendation text
    """
    # Format retrieved songs for the prompt
    song_descriptions = []
    for i, (song, sim_score) in enumerate(retrieved_songs, 1):
        desc = (
            f'{i}. "{song["title"]}" by {song["artist"]} - '
            f'{song["genre"]}, {song["mood"]}, '
            f'energy: {song["energy"]:.2f}, tempo: {song["tempo_bpm"]} BPM, '
            f'acousticness: {song["acousticness"]:.2f} '
            f'(relevance: {sim_score:.3f})'
        )
        song_descriptions.append(desc)

    user_message = (
        f'User request: "{query}"\n\n'
        f'Retrieved songs:\n' + "\n".join(song_descriptions) +
        f'\n\nRecommend top {k} songs.'
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(k=k)},
            {"role": "user", "content": FEW_SHOT_USER},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=800,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 5. Convenience: Full RAG Pipeline
# ---------------------------------------------------------------------------

def rag_recommend(
    query: str,
    songs: List[Dict],
    song_embeddings: List[List[float]],
    top_k_retrieve: int = 10,
    top_k_recommend: int = 3,
) -> Dict:
    """
    Full RAG pipeline: retrieve relevant songs, then generate recommendations.

    Args:
        query: User's natural language request
        songs: Full song catalog
        song_embeddings: Pre-computed embeddings
        top_k_retrieve: Number of songs to retrieve from vector search
        top_k_recommend: Number of songs to recommend in final output

    Returns:
        Dictionary with 'query', 'retrieved_songs', and 'recommendation' keys
    """
    # Step 1: Retrieve
    retrieved = retrieve_songs(query, songs, song_embeddings, top_k=top_k_retrieve)

    # Step 2: Generate
    recommendation = generate_recommendation(query, retrieved, k=top_k_recommend)

    return {
        "query": query,
        "retrieved_songs": [(s["title"], s["artist"], round(score, 3)) for s, score in retrieved],
        "recommendation": recommendation,
    }
