from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv
    songs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # csv.DictReader reads every value as a string.
            # Numeric fields must be cast so math operations work later.
            songs.append({
                "id":           int(row["id"]),       # integer
                "title":        row["title"],          # string
                "artist":       row["artist"],         # string
                "genre":        row["genre"],          # string
                "mood":         row["mood"],           # string
                "energy":       float(row["energy"]),       # 0.0 – 1.0
                "tempo_bpm":    float(row["tempo_bpm"]),    # 60 – 168
                "valence":      float(row["valence"]),      # 0.0 – 1.0
                "danceability": float(row["danceability"]), # 0.0 – 1.0
                "acousticness": float(row["acousticness"]), # 0.0 – 1.0
            })
    return songs

MOOD_NEIGHBORS = {
    "chill":      {"relaxed", "focused", "dreamy"},
    "relaxed":    {"chill", "focused", "nostalgic"},
    "focused":    {"chill", "relaxed"},
    "happy":      {"energetic", "euphoric", "uplifting", "playful"},
    "energetic":  {"happy", "intense", "euphoric"},
    "intense":    {"energetic", "angry"},
    "sad":        {"melancholy", "moody"},
    "melancholy": {"sad", "moody"},
    "moody":      {"sad", "melancholy"},
    "romantic":   {"relaxed", "dreamy"},
    "euphoric":   {"happy", "energetic", "uplifting"},
    "nostalgic":  {"relaxed", "melancholy"},
    "angry":      {"intense"},
    "uplifting":  {"happy", "euphoric", "playful"},
    "playful":    {"happy", "uplifting"},
    "dreamy":     {"chill", "romantic"},
}

MOOD_TO_VALENCE = {
    "happy":      0.80,
    "energetic":  0.70,
    "chill":      0.58,
    "focused":    0.60,
    "relaxed":    0.70,
    "intense":    0.40,
    "sad":        0.25,
    "melancholy": 0.25,
    "moody":      0.35,
    "romantic":   0.72,
    "euphoric":   0.90,
    "nostalgic":  0.50,
    "angry":      0.20,
    "uplifting":  0.82,
    "playful":    0.85,
    "dreamy":     0.62,
}

RELATED_GENRES = {
    "pop":       {"indie pop", "dance-pop", "dream pop", "synthwave"},
    "indie pop": {"pop", "dream pop"},
    "dance-pop": {"pop", "edm"},
    "dream pop": {"pop", "indie pop", "ambient"},
    "lofi":      {"ambient", "chillhop", "jazz"},
    "ambient":   {"lofi", "dream pop", "classical"},
    "chillhop":  {"lofi", "hip hop"},
    "rock":      {"metal", "country"},
    "metal":     {"rock"},
    "country":   {"rock", "soul", "reggae"},
    "hip hop":   {"r&b", "soul", "funk"},
    "r&b":       {"soul", "hip hop", "funk"},
    "soul":      {"r&b", "funk", "hip hop"},
    "funk":      {"soul", "r&b", "hip hop"},
    "jazz":      {"lofi", "soul", "r&b"},
    "edm":       {"synthwave", "dance-pop"},
    "synthwave": {"edm", "pop"},
    "reggae":    {"country", "soul"},
    "classical": {"ambient"},
}

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Return a (0.0-1.0) score and a list of reason strings for one song."""
    reasons = []

    # Genre Score (35%)
    user_genre = user_prefs.get("favorite_genre", "")
    song_genre = song.get("genre", "")
    if song_genre == user_genre:
        genre_score = 1.0
    elif song_genre in RELATED_GENRES.get(user_genre, set()):
        genre_score = 0.5
    else:
        genre_score = 0.0

    # Mood Score (25%)
    user_mood = user_prefs.get("favorite_mood", "")
    song_mood = song.get("mood", "")
    if song_mood == user_mood:
        mood_score = 1.0
    elif song_mood in MOOD_NEIGHBORS.get(user_mood, set()):
        mood_score = 0.5
    else:
        mood_score = 0.0

    # Energy Score (20%)
    target_energy = user_prefs.get("target_energy", 0.5)
    energy_score = 1.0 - abs(song.get("energy", 0.5) - target_energy)

    # Acoustic Score (10%)
    acousticness = song.get("acousticness", 0.5)
    if user_prefs.get("likes_acoustic", False):
        acoustic_score = acousticness
    else:
        acoustic_score = 1.0 - acousticness

    # Tempo Score (5%)
    target_bpm = 60 + (target_energy * 108)
    song_tempo_norm   = (song.get("tempo_bpm", 114) - 60) / 108
    target_tempo_norm = (target_bpm - 60) / 108
    tempo_score = 1.0 - abs(song_tempo_norm - target_tempo_norm)

    # Valence Score (5%)
    expected_valence = MOOD_TO_VALENCE.get(user_mood, 0.60)
    valence_score = 1.0 - abs(song.get("valence", 0.5) - expected_valence)

    # Weighted total
    total_score = (
          genre_score    * 0.35
        + mood_score     * 0.25
        + energy_score   * 0.20
        + acoustic_score * 0.10
        + tempo_score    * 0.05
        + valence_score  * 0.05
    )

    # Reasons
    if genre_score    >= 0.7: reasons.append("matches your favorite genre")
    if mood_score     >= 0.7: reasons.append("matches your mood")
    if energy_score   >= 0.7: reasons.append("energy level close to your preference")
    if acoustic_score >= 0.7: reasons.append("fits your acoustic preference")
    if tempo_score    >= 0.6: reasons.append("tempo matches your vibe")
    if valence_score  >= 0.6: reasons.append("emotional tone aligns with your taste")

    return (total_score, reasons)

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song, then return the top k sorted by score descending."""
    scored = [
        (song, score, ", ".join(reasons) or "no specific reasons")
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
