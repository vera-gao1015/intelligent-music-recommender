"""
Streamlit Web UI for the Intelligent Music Recommender.

Chat-style interface for natural music recommendations.

Usage:
    streamlit run app.py
"""

import streamlit as st
from src.recommender import load_songs
from src.rag import build_song_embeddings, retrieve_songs, generate_recommendation
from src.agent import run_agent
from src.guardrails import validate_input, run_guardrails

# Page config
st.set_page_config(
    page_title="Intelligent Music Recommender",
    page_icon="🎵",
    layout="centered",
)

# Custom CSS for chat styling
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    .agent-step {
        padding: 6px 12px;
        margin: 4px 0;
        border-left: 3px solid #4A90D9;
        background-color: rgba(74, 144, 217, 0.05);
        border-radius: 0 6px 6px 0;
        font-size: 0.9em;
    }
    .guardrail-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 500;
        margin: 2px 4px;
    }
    .badge-pass {
        background-color: rgba(40, 167, 69, 0.15);
        color: #28a745;
    }
    .badge-fail {
        background-color: rgba(220, 53, 69, 0.15);
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """Load songs and build embeddings once, then cache."""
    songs = load_songs("data/songs.csv")
    embeddings = build_song_embeddings(songs)
    return songs, embeddings


# Load system
songs, embeddings = load_system()

# Header
st.title("🎵 Intelligent Music Recommender")
st.caption("Powered by RAG + Agentic Workflow + GPT-4o-mini")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ---------- Name Entry Gate ----------
if not st.session_state.user_name:
    st.markdown("### Welcome! What's your name?")
    name_input = st.text_input(
        "Enter your name to get started",
        placeholder="e.g., Alex",
        label_visibility="collapsed",
    )
    if st.button("Let's Go!", type="primary") and name_input.strip():
        st.session_state.user_name = name_input.strip()
        st.rerun()
    st.stop()

# ---------- Main Chat Interface ----------

# Show welcome message if no history
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🎵"):
        st.markdown(
            f"Hey **{st.session_state.user_name}**! I'm your music recommender. "
            f"I have **{len(songs)} songs** across 17 genres ready to go.\n\n"
            "Tell me how you're feeling or what you're doing, "
            "and I'll find the perfect tracks for you."
        )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"], unsafe_allow_html=True)
        # Re-render expanders for assistant messages
        if msg["role"] == "assistant" and "agent_steps" in msg:
            with st.expander("🔍 Agent Workflow", expanded=False):
                st.markdown(msg["agent_steps"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and "guardrail_html" in msg:
            with st.expander("🛡️ Guardrail Report", expanded=False):
                st.markdown(msg["guardrail_html"], unsafe_allow_html=True)

# Chat input
if query := st.chat_input(f"What kind of music are you looking for, {st.session_state.user_name}?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Validate input
    is_valid, validation_msg = validate_input(query)

    if not is_valid:
        error_content = f"⚠️ {validation_msg}\n\nPlease describe what kind of music you'd like!"
        with st.chat_message("assistant", avatar="🎵"):
            st.markdown(error_content)
        st.session_state.messages.append({
            "role": "assistant",
            "avatar": "🎵",
            "content": error_content,
        })
    else:
        # Show thinking indicator and run agent
        with st.chat_message("assistant", avatar="🎵"):
            with st.spinner("🎧 Finding your perfect tracks..."):
                result = run_agent(
                    query=query,
                    songs=songs,
                    song_embeddings=embeddings,
                    rag_retrieve_fn=retrieve_songs,
                    rag_generate_fn=generate_recommendation,
                    verbose=False,
                )

            # Build recommendation content
            recommendation_text = result["recommendation"]
            st.markdown(recommendation_text)

            # Build agent steps HTML
            agent_lines = []
            for step in result["steps"]:
                if step["step"] == 1:
                    agent_lines.append(
                        f'<div class="agent-step"><b>Step 1 — Intent:</b> '
                        f'{step["result"]}<br><i>{step["reasoning"]}</i></div>'
                    )
                elif step["step"] == 2:
                    agent_lines.append(
                        f'<div class="agent-step"><b>Step 2 — Strategy:</b> '
                        f'{step["strategy_name"]}<br><i>{step["description"]}</i></div>'
                    )
                elif step["step"] == 3:
                    matches_html = "".join(
                        f'<br>• "{m["title"]}" by {m["artist"]} '
                        f'(match score: {m["match_score"]})'
                        for m in step["top_matches"]
                    )
                    agent_lines.append(
                        f'<div class="agent-step"><b>Step 3 — Retrieved '
                        f'{step["songs_retrieved"]} candidates</b>{matches_html}</div>'
                    )
                elif step["step"] == 4:
                    agent_lines.append(
                        f'<div class="agent-step"><b>Step 4 — Generated recommendation</b> '
                        f'({step["output_length"]} chars)</div>'
                    )
            agent_steps_html = "".join(agent_lines)

            with st.expander("🔍 Agent Workflow", expanded=False):
                st.markdown(agent_steps_html, unsafe_allow_html=True)

            # Guardrail report
            retrieved = retrieve_songs(query, songs, embeddings, top_k=10)
            report = run_guardrails(query, result["recommendation"], retrieved)

            def badge(label, passed):
                cls = "badge-pass" if passed else "badge-fail"
                icon = "✅" if passed else "❌"
                return f'<span class="guardrail-badge {cls}">{icon} {label}</span>'

            guardrail_html = (
                badge("Input Validation", report["input_validation"]["passed"])
                + badge("Output Verification", report["output_verification"]["passed"])
                + badge("Quality Check", report["quality_check"]["passed"])
            )

            warnings = report["output_verification"].get("warnings", [])
            issues = report["quality_check"].get("issues", [])
            if warnings or issues:
                guardrail_html += "<br><br>"
                for w in warnings:
                    guardrail_html += f'<div style="color: #856404; background: #fff3cd; padding: 6px 10px; border-radius: 6px; margin: 4px 0;">⚠️ {w}</div>'
                for issue in issues:
                    guardrail_html += f'<div style="color: #856404; background: #fff3cd; padding: 6px 10px; border-radius: 6px; margin: 4px 0;">⚠️ {issue}</div>'

            with st.expander("🛡️ Guardrail Report", expanded=False):
                st.markdown(guardrail_html, unsafe_allow_html=True)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "avatar": "🎵",
            "content": recommendation_text,
            "agent_steps": agent_steps_html,
            "guardrail_html": guardrail_html,
        })
