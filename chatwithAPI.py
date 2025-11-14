# chat.py
import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from google.api_core.exceptions import ResourceExhausted
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 👉 RAG RETRIEVER
from retriever import vector_search

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables!")

# Initialize Gemini model with retry configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_retries=2  # Limit retries to avoid long waits
)

# In-memory session storage
sessions = {}

def fetch_api_data():
    """Call external API once and return JSON data."""
    try:
        response = requests.get("https://api.escuelajs.co/api/v1/products", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data[:5]  # take first 5 items to keep context small
    except requests.exceptions.RequestException as e:
        logger.error(f"API fetch error: {str(e)}")
        return [{"error": f"Failed to fetch product data: {str(e)}"}]
    except Exception as e:
        logger.error(f"Unexpected error in fetch_api_data: {str(e)}")
        return [{"error": str(e)}]

def get_or_create_session(session_id: str):
    """Create a new session and fetch API data if not exists."""
    if session_id not in sessions:
        api_data = fetch_api_data()

        system_context = (
            "You are an AI assistant for NITRO LINE Automobile Shop. "
            "Answer user questions using the PDF knowledge base and external product API data provided.\n"
            "Below is product data fetched from an external API:\n"
            f"{api_data}\n"
            "Use this data only when relevant to the question.\n"
        )

        sessions[session_id] = {
            "history": [SystemMessage(content=system_context)],
            "api_data": api_data
        }

    return sessions[session_id]

def chat(session_id: str, user_input: str) -> str:
    """Main chat function with session memory, RAG, and API data."""

    session = get_or_create_session(session_id)

    # -----------------------------
    # 🔥 Step 1: Perform RAG Search
    # -----------------------------
    rag_results = vector_search(user_input)

    if rag_results:
        # Format RAG results as plain text for the model
        rag_text = "\n".join([f"- {r['text']}" for r in rag_results[:5]])  # top 5
    else:
        rag_text = "No relevant PDF content found."

    # -----------------------------
    # 🔥 Step 2: Combine Inputs
    # -----------------------------
    final_prompt = f"""
Use the following information when generating your answer:

1️⃣ **Relevant PDF Knowledge (RAG Search Results):**
{rag_text}

2️⃣ **External API Product Data:**
{session['api_data']}

Now answer the following question naturally and clearly:

❓ **User Question:** {user_input}
"""

    session["history"].append(HumanMessage(content=final_prompt))

    # -----------------------------
    # 🔥 Step 3: Generate Response
    # -----------------------------
    ai_msg: AIMessage = llm.invoke(session["history"])
    session["history"].append(ai_msg)

    return ai_msg.content
