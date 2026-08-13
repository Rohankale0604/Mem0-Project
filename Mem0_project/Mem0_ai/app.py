"""
Streamlit frontend for the Mem0 + LangChain Interview Prep Coach.
"""

import streamlit as st

from memory_agent import chat, get_all_memories


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Interview Prep Coach",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_memories" not in st.session_state:
    st.session_state.show_memories = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Interview Coach")

    st.caption(
        "A memory-aware AI assistant for "
        "technical interview preparation."
    )

    st.divider()

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    st.subheader("👤 Candidate")

    user_id = st.text_input(
        "User ID",
        value="Rohan",
        help="Mem0 uses this ID to keep each candidate's memories separate.",
    ).strip()

    if not user_id:
        user_id = "Rohan"

    st.divider()

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    st.subheader("🧠 Memory")

    if st.button(
        "View Stored Memories",
        use_container_width=True,
    ):
        st.session_state.show_memories = True

    if st.button(
        "Hide Memories",
        use_container_width=True,
    ):
        st.session_state.show_memories = False

    if st.session_state.show_memories:

        memory_data = get_all_memories(user_id)

        if memory_data.get("error"):

            st.error(memory_data["error"])

        else:

            memories = memory_data.get("results", [])

            if memories:

                st.markdown("#### What I remember")

                for memory in memories:

                    if isinstance(memory, dict):
                        memory_text = memory.get(
                            "memory",
                            "Unknown memory",
                        )
                    else:
                        memory_text = str(memory)

                    st.info(
                        f"🧠 {memory_text}"
                    )

            else:

                st.info(
                    "No memories stored yet. "
                    "Start chatting to build your profile."
                )

    st.divider()

    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

    st.subheader("💬 Conversation")

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # Technology
    # --------------------------------------------------------

    st.subheader("⚙️ Technology")

    st.write("🧠 Mem0")
    st.write("🔗 LangChain")
    st.write("⚡ Groq")
    st.write("💎 Gemini Embeddings")
    st.write("🗄️ Qdrant")
    st.write("🎈 Streamlit")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🧠 AI Interview Prep Coach")

st.write(
    "Your personalized interview coach that remembers "
    "your preparation, skills, previous discussions, "
    "and areas for improvement."
)

st.divider()


# ============================================================
# WELCOME SECTION
# ============================================================

if not st.session_state.messages:

    st.info(
        "👋 Welcome! Start a conversation below to begin "
        "your interview preparation."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("🎯 Practice")

        st.write(
            "Practice Python, SQL, Machine Learning, "
            "Generative AI, LangChain and other "
            "technical interview topics."
        )

    with col2:

        st.subheader("🧠 Remember")

        st.write(
            "Mem0 remembers useful information from "
            "your previous conversations and preparation."
        )

    with col3:

        st.subheader("📈 Improve")

        st.write(
            "Receive personalized questions and feedback "
            "based on your previous preparation."
        )

    st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask a question or practice an interview answer..."
)


if user_input:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)

    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Checking memories and preparing your response..."
        ):

            reply = chat(
                user_id=user_id,
                user_message=user_input,
            )

        st.markdown(reply)

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Mem0 • LangChain • Groq • Gemini • Qdrant • Streamlit"
)