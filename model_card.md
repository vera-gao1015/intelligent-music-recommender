# Model Card: Intelligent Music Recommender

## 1. Model Name

**Intelligent Music Recommender v2.0**

---

## 2. Intended Use

This system is designed to recommend songs from a curated catalog based on a user's natural language description of their mood, activity, or taste. Unlike the original rule-based version, this system uses AI (RAG, agentic reasoning, and LLM generation) to understand open-ended requests and produce personalized, explained recommendations.

It is built as an educational project demonstrating applied AI techniques. It is not intended for production music streaming services.

**Not intended for:** real-time music apps, catalogs with millions of songs, or users expecting real Spotify/Apple Music integration.

---

## 3. How the System Works

The system operates as a multi-step AI pipeline:

1. **Input Guardrail** — Validates user input for length, content, and safety (blocks prompt injection attempts).
2. **Agent (Intent Classification)** — An LLM classifies the user's request into one of three intents: "recommend" (mood-based), "explore" (genre discovery), or "compare" (contrasting styles).
3. **Strategy Selection** — Based on the intent, the agent picks a strategy that determines how many songs to retrieve and recommend.
4. **RAG Retrieval** — Each song's metadata is converted to a text description and embedded using OpenAI's `text-embedding-3-small` model. The user's query is also embedded, and cosine similarity identifies the most relevant songs.
5. **LLM Generation** — The retrieved songs and user query are sent to GPT-4o-mini with few-shot examples. The model generates personalized explanations for why each song fits.
6. **Output Guardrail** — Checks that the LLM only references songs that were actually retrieved (anti-hallucination), and verifies response quality (correct count, sufficient detail).

The original rule-based scoring system from Project 3 is preserved in `recommender.py` and can still be used independently.

---

## 4. Data

- The catalog has **50 songs** stored in a CSV file (expanded from the original 20).
- Songs cover **17 genres**: pop, rock, lofi, ambient, jazz, hip hop, r&b, soul, funk, edm, synthwave, classical, country, reggae, metal, indie pop, dream pop.
- **16 moods** are represented: happy, chill, intense, sad, relaxed, focused, romantic, dreamy, angry, moody, euphoric, nostalgic, uplifting, playful, melancholy, energetic.
- Each song has 10 attributes: id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness.
- The dataset was constructed by hand with fictional artists and titles. It is not based on real streaming data.
- **Limits:** 50 songs is still small. Real recommenders work with millions. Some genres have only 2 songs, which limits recommendation diversity for niche tastes.

---

## 5. Strengths

- **Natural language understanding** — Users can describe what they want in their own words instead of selecting from menus.
- **Semantic search** — RAG retrieval finds songs based on meaning, not just keyword matching. "I'm tired and want to unwind" correctly retrieves low-energy ambient and lofi tracks.
- **Adaptive strategies** — The agent automatically adjusts behavior: mood-based requests get focused recommendations, while exploration requests return diverse genre picks.
- **Transparent reasoning** — Every intermediate step (intent, strategy, retrieved candidates, guardrail results) is logged and displayed, making the system's decisions fully observable.
- **Anti-hallucination** — The output guardrail catches cases where the LLM invents songs that don't exist in the catalog.

---

## 6. Limitations and Bias

**Inherited bias from the original system:** The rule-based scoring still weights genre at 35%, which means the RAG retrieval can sometimes be overridden by genre dominance when the two systems are compared.

**Embedding bias:** The OpenAI embedding model was trained on general text, not music-specific data. This means it may not perfectly capture the nuances of musical similarity. For example, "lofi" and "chillhop" are musically very similar, but the embedding model may not place them as close as a music-specific model would.

**Small catalog bias:** With only 50 songs, some user queries may not have truly good matches. The system will still return its best candidates, which might not perfectly fit the request.

**LLM bias:** GPT-4o-mini may favor certain genres or use language patterns that reflect its training data biases. The few-shot examples help constrain this, but the model might still describe certain genres more enthusiastically than others.

**Language bias:** The system currently works best with English-language queries. Non-English input may produce lower-quality intent classification and recommendations.

---

## 7. Evaluation

**Automated testing:** The evaluation suite (`eval/evaluate.py`) runs 19 tests across four categories:
- **Input Validation (9 tests):** Verifies that valid queries pass and invalid ones (empty, too short, prompt injection) are correctly rejected. All 9 passed.
- **Intent Classification (4 tests):** Checks that the agent correctly identifies recommend, explore, and compare intents. All 4 passed.
- **Retrieval Relevance (3 tests):** Confirms that RAG retrieval returns genre/mood-appropriate songs in the top 3 results. All 3 passed.
- **End-to-End (3 tests):** Runs the full pipeline and verifies output passes all guardrail checks. All 3 passed.

**Pass rate: 100% (19/19)**

**Manual testing observations:**
- For "I feel tired after work and want to relax," the system consistently retrieves low-energy ambient and lofi tracks — the results feel natural and appropriate.
- For "I want to discover new genres," the system returns songs across 5 different genres, demonstrating good diversity.
- The anti-hallucination guardrail has not triggered during testing, suggesting the few-shot prompt effectively constrains the LLM to reference only retrieved songs.

---

## 8. Future Work

- **User feedback loop:** Allow users to rate recommendations (thumbs up/down) and use this data to fine-tune retrieval weights over time.
- **Larger catalog:** Expand beyond 50 songs to make the system more useful and test scalability.
- **Music-specific embeddings:** Replace general-purpose OpenAI embeddings with a music-specific model for better semantic matching.
- **Hybrid scoring:** Combine the original rule-based scores with RAG similarity scores for a weighted hybrid approach.
- **Streamlit UI:** Build a web interface for a more user-friendly experience (the system is already architected to support this).
- **Multi-turn conversation:** Let users refine recommendations through follow-up messages ("more like #2 but with higher energy").

---

## 9. Reflection on AI Collaboration

**How AI helped during development:**
- AI was used to generate the expanded song dataset (30 additional songs with consistent formatting).
- AI assisted in designing the system architecture and identifying which rubric requirements each module would satisfy.
- AI helped write boilerplate code for embedding caching, logging setup, and test case structures.

**One helpful AI suggestion:** When designing the agent module, AI suggested classifying user intent into three categories (recommend/explore/compare) rather than just recommend vs. not-recommend. This made the system significantly more useful because it naturally adapts its behavior — exploration queries return more diverse results with more songs.

**One flawed AI suggestion:** Initially, the AI suggested using a complex chain-of-thought prompt for intent classification. In practice, this was slower, more expensive, and occasionally produced inconsistent JSON output. Switching to a simple, constrained prompt with `temperature=0` fixed the reliability issue.

**System limitations I recognize:**
- The 50-song catalog is too small for a real recommendation system. Users with very specific tastes may not find satisfying matches.
- The system makes a new API call for each query's embedding, which adds latency. A production system would need query caching.
- The intent classifier sometimes misclassifies single-word inputs (e.g., "jazz" was classified as "explore" rather than "recommend"), though this is arguably a reasonable interpretation.

**Could the AI be misused, and how do we prevent it?**
The main misuse risk is prompt injection — someone could try to trick the LLM into ignoring its instructions and producing unrelated or harmful content. We prevent this with the input guardrail, which detects and blocks common injection patterns like "ignore previous instructions" before they reach the LLM. Another risk is that users might trust the AI's recommendations too much, especially if they assume the system knows more than it does. In reality, it only searches a 50-song catalog. We mitigate this by keeping the system transparent — every step is logged and the guardrail report is shown after each recommendation.

**What surprised me while testing reliability?**
I was surprised that the anti-hallucination guardrail never triggered during testing. I expected the LLM to occasionally invent song titles, but the few-shot prompting constrained it well enough that it always stuck to retrieved songs. On the other hand, I was surprised that the intent classifier turned single-word inputs like "jazz" into "explore" — technically reasonable, but not what most users would expect.

**Ethical considerations:**
- The system does not collect or store any user data beyond the current session.
- The guardrails prevent prompt injection, protecting against misuse.
- All song data is fictional, so there are no copyright concerns.
- The system's recommendations could create a "filter bubble" if always used in recommend mode — the explore mode helps mitigate this by intentionally surfacing diverse content.
