import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("❌ MONGO_URI not found in environment variables!")

client = MongoClient(MONGO_URI)
db = client["vector-db"]
collection = db["docs"]

# HuggingFace Embedding
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str):
    return embedding_model.encode(text).tolist()

def insert_document(content: str, metadata: dict = None):
    vector = embed_text(content)
    doc = {
        "content": content,
        "embedding": vector,
        "metadata": metadata or {}
    }
    return collection.insert_one(doc)

def get_collection():
    return collection
