# rag/ingest.py
import os
from dotenv import load_dotenv
from db import get_collection, embed_text  # use HF embeddings
from loader import load_and_chunk_pdf

load_dotenv()

collection = get_collection()

def ingest_pdf(pdf_path: str):
    chunks = load_and_chunk_pdf(pdf_path)

    for doc in chunks:
        embedding = embed_text(doc.page_content)  # HuggingFace embedding

        record = {
            "text": doc.page_content,
            "embedding": embedding,  # note: MongoDB vector field is "embedding"
            "metadata": doc.metadata
        }

        collection.insert_one(record)

    print("✅ PDF successfully ingested into MongoDB!")

if __name__ == "__main__":
    ingest_pdf("nitro.pdf")
