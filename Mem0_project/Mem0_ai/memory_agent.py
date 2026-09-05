"""
Memory layer for the Mem0 + LangChain Interview Prep Coach.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from mem0 import Memory

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"
QDRANT_PATH = PROJECT_ROOT / "qdrant_storage"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from the .env file."
    )

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing from the .env file."
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# MEM0 CONFIGURATION
# ============================================================

MEM0_CONFIG = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": GROQ_MODEL,
            "api_key": GROQ_API_KEY,
            "temperature": 0,
            "max_tokens": 300,
        },
    },

    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/gemini-embedding-001",
            "embedding_dims": 768,
            "api_key": GOOGLE_API_KEY,
        },
    },

   "vector_store": {
    "provider": "qdrant",
    "config": {
        "collection_name": "mem0_interview_prep",
        "embedding_model_dims": 768,
        "url": os.getenv("QDRANT_URL"),
        "api_key": os.getenv("QDRANT_API_KEY"),
    },
},
      
}


# ============================================================
# INITIALIZE MEM0
# ============================================================

try:
    MEMORY = Memory.from_config(MEM0_CONFIG)
    MEMORY_INIT_ERROR = None

except Exception as exc:
    MEMORY = None
    MEMORY_INIT_ERROR = exc


# ============================================================
# LANGCHAIN INTERVIEW COACH PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI Interview Preparation Coach.

Help the candidate prepare for technical interviews.

Use the candidate's memories below to personalize your response.

Candidate memories:
{memories}

Instructions:

1. Personalize the response using relevant memories.
2. Avoid unnecessarily repeating topics already covered.
3. Focus on areas where the candidate needs improvement.
4. Ask practical interview questions.
5. If the candidate gives an answer, evaluate it.
6. Give concise and specific feedback.
7. Be professional and helpful.
8. Do not mention the internal memory system.

Current candidate message:
{question}
"""


# ============================================================
# LANGCHAIN CHAIN
# ============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


LLM = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.3,
    api_key=GROQ_API_KEY,
)


CHAIN = PROMPT | LLM | StrOutputParser()


# ============================================================
# FORMAT MEMORIES
# ============================================================

def format_memories(search_results):
    """
    Convert Mem0 search results into readable text.
    """

    if isinstance(search_results, dict):
        items = search_results.get("results", [])
    else:
        items = search_results

    memories = []

    for item in items:

        if isinstance(item, dict):

            memory_text = item.get("memory")

            if memory_text:
                memories.append(
                    f"- {memory_text}"
                )

    if memories:
        return "\n".join(memories)

    return "(No prior memories found.)"


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(
    user_id: str,
    user_message: str,
):
    """
    Store the user's message directly in Mem0.

    infer=False prevents Mem0 from sending the message
    to the LLM for memory extraction.
    """

    if MEMORY is None:
        return None

    try:

        result = MEMORY.add(
            [
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            user_id=user_id,
            infer=False,
        )

        print()
        print("=" * 60)
        print("MEM0 ADD RESULT:")
        print(result)
        print("=" * 60)
        print()

        return result

    except Exception as exc:

        print()
        print("=" * 60)
        print("MEM0 MEMORY SAVE ERROR:")
        print(exc)
        print("=" * 60)
        print()

        return None


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(
    user_id: str,
    user_message: str,
) -> str:
    """
    1. Search relevant memories.
    2. Generate AI response.
    3. Save the user's message to Mem0.
    """

    if MEMORY is None:

        return (
            "Mem0 initialization failed.\n\n"
            f"Error: {MEMORY_INIT_ERROR}"
        )

    try:

        # ----------------------------------------------------
        # SEARCH RELEVANT MEMORIES
        # ----------------------------------------------------

        search_results = MEMORY.search(
            query=user_message,
            filters={
                "user_id": user_id
            },
            limit=3,
        )

        memories_text = format_memories(
            search_results
        )

        # ----------------------------------------------------
        # GENERATE AI RESPONSE
        # ----------------------------------------------------

        reply = CHAIN.invoke(
            {
                "memories": memories_text,
                "question": user_message,
            }
        )

        # ----------------------------------------------------
        # SAVE USER MESSAGE TO MEM0
        # ----------------------------------------------------

        save_memory(
            user_id=user_id,
            user_message=user_message,
        )

        return reply

    except Exception as exc:

        return (
            "I encountered an error while processing "
            "your request.\n\n"
            f"Error: {exc}"
        )


# ============================================================
# GET ALL MEMORIES
# ============================================================

def get_all_memories(user_id: str):

    if MEMORY is None:

        return {
            "error": str(MEMORY_INIT_ERROR),
            "results": [],
        }

    try:

        return MEMORY.get_all(
            filters={
                "user_id": user_id
            }
        )

    except Exception as exc:

        return {
            "error": str(exc),
            "results": [],
        }
