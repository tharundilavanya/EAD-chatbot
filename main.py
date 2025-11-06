# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatwithAPI import chat

app = FastAPI(
    title="Automobile Shop AI Assistant",
    description="FastAPI backend for Gemini-powered automobile assistant",
    version="1.0.0"
)

# Request body model
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        response = chat(request.session_id, request.user_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the Automobile Shop AI API!"}
