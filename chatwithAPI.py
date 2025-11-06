import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables!")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
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
    except Exception as e:
        return [{"error": str(e)}]

def get_or_create_session(session_id: str):
    """Create a new session and fetch API data if not exists."""
    if session_id not in sessions:
        api_data = fetch_api_data()
        system_context = (
            "You are a helpful AI assistant for a automobile shop named Your automobile shop "
            "Your Automobile Shop is a Trusted Vehicle Care in Katubedda. "
            "Since our humble beginnings, Your Automobile Shop has grown into one of Katubedda’s most trusted and reliable automotive service centers. "
            "What started as a small local workshop built on passion and precision has evolved into a full-scale automobile hub dedicated to keeping your vehicle in perfect condition. "
            "We’ve been helping drivers get back on the road with confidence. "
            "From routine maintenance to major repairs, our expert team of certified technicians delivers professional, honest, and efficient service every time. "
            "We combine modern diagnostic tools with time-tested expertise, ensuring your car receives the care it deserves. "
            "We take pride in our transparent pricing, quick turnaround times, and friendly customer service. "
            "Our services include full vehicle servicing and maintenance, engine and transmission repairs, brake, suspension and steering system overhauls, "
            "wheel alignment and balancing, vehicle detailing and body painting, and battery, tyre, and genuine spare parts replacement. "
            "We offer skilled and certified mechanics, the latest diagnostic technology, honest pricing with no hidden costs, and excellent after-service support. "
            "Your Automobile Shop is located in Katubedda, open Monday to Saturday from 8:00 AM to 7:00 PM. "
            "Contact us by phone at 0111042838, WhatsApp 0771042838, or email yourautomobileshop@gmail.com."
            "You have access to the following product data from an external API:\n"
            f"{api_data}\n"
            "Use this data when answering relevant questions."
        )

        sessions[session_id] = {
            "history": [SystemMessage(content=system_context)],
            "api_data": api_data
        }
    return sessions[session_id]

def chat(session_id: str, user_input: str) -> str:
    """Main chat function with context memory and API data."""
    session = get_or_create_session(session_id)
    session["history"].append(HumanMessage(content=user_input))

    ai_msg: AIMessage = llm.invoke(session["history"])
    session["history"].append(ai_msg)

    return ai_msg.content

# if __name__ == "__main__":
#     print("🤖 Gemini Chat with API Context")
#     session_id = input("Enter session ID: ").strip()

#     while True:
#         user_input = input("You: ").strip()
#         if user_input.lower() in {"exit", "quit"}:
#             print("Goodbye! 👋")
#             break

#         response = chat(session_id, user_input)
#         print("Assistant:", response)

