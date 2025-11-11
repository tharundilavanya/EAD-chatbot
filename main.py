# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from chatwithAPI import chat
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Automobile Shop AI Assistant",
    description="FastAPI backend for Gemini-powered automobile assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request body model
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

# Response models for better error handling
class ChatResponse(BaseModel):
    response: str
    
class ErrorResponse(BaseModel):
    error: str
    error_code: str
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        if not request.session_id or not request.user_input:
            raise HTTPException(
                status_code=400, 
                detail="Both session_id and user_input are required"
            )
        
        response = chat(request.session_id, request.user_input)
        return {"response": response}
    
    except ValueError as ve:
        # Handle our custom errors from chatwithAPI.py
        error_msg = str(ve)
        logger.error(f"ValueError in chat_endpoint: {error_msg}")
        
        if "QUOTA_EXCEEDED" in error_msg:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "error_code": "QUOTA_EXCEEDED",
                    "message": "Our AI service is currently experiencing high demand. Please try again in a minute."
                }
            )
        elif "AI_ERROR" in error_msg:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service unavailable",
                    "error_code": "AI_ERROR",
                    "message": "Unable to process your request at this time. Please try again later."
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_code": "SYSTEM_ERROR",
                    "message": error_msg.replace("SYSTEM_ERROR: ", "")
                }
            )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error in chat_endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_code": "UNEXPECTED_ERROR",
                "message": "An unexpected error occurred. Please try again later."
            }
        )

@app.get("/")
def root():
    return {"message": "Welcome to the Automobile Shop AI API!"}
