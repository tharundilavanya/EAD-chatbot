# # # retriever.py
# # from db import get_collection, embed_text

# # collection = get_collection()

# # def vector_search(query: str, k: int = 3):
# #     query_vector = embed_text(query)

# #     pipeline = [
# #         {
# #             "$vectorSearch": {
# #                 "queryVector": query_vector,
# #                 "path": "vector",
# #                 "numCandidates": 50,
# #                 "limit": k,
# #                 "index": "vector_index"
# #             }
# #         },
# #         {
# #             "$project": {
# #                 "_id": 0,
# #                 "content": 1,
# #                 "score": {"$meta": "vectorSearchScore"}
# #             }
# #         }
# #     ]

# #     results = list(collection.aggregate(pipeline))
# #     return results
# from db import get_collection, embed_text

# collection = get_collection()

# def vector_search(query: str, k: int = 3):
#     query_vector = embed_text(query)

#     pipeline = [
#         {
#             "$vectorSearch": {
#                 "queryVector": query_vector,
#                 "path": "vector",
#                 "numCandidates": 50,
#                 "limit": k,
#                 "index": "law_docs_index"  # make sure this matches Atlas
#             }
#         },
#         {
#             "$project": {
#                 "_id": 0,
#                 "text": 1,  # match the field in your DB
#                 "score": {"$meta": "vectorSearchScore"}
#             }
#         }
#     ]

#     results = list(collection.aggregate(pipeline))
#     return results

# rag/retriever.py
from db import get_collection, embed_text  # HuggingFace embeddings

collection = get_collection()

def vector_search(query: str, k: int = 3):
    """
    Perform a vector search in MongoDB using the HuggingFace embedding model.
    
    Args:
        query (str): User query.
        k (int): Number of top results to return.
    
    Returns:
        List of dicts containing 'text', 'metadata', and 'score'.
    """
    try:
        # 1️⃣ Embed the query
        query_vector = embed_text(query)

        # 2️⃣ MongoDB Atlas $vectorSearch pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_vector,
                    "path": "embedding",   # Must match field in MongoDB Atlas index
                    "numCandidates": 50,   # How many candidates to consider
                    "limit": k,
                    "index": "law_docs_index" # Name of the vector index in Atlas
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,   # Field containing the original text
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # 3️⃣ Execute aggregation
        results = list(collection.aggregate(pipeline))
        print(f"✅ Vector search returned {len(results)} results for query: '{query}'")
        return results
        
    except Exception as e:
        print(f"❌ Vector search error: {e}")
        print("⚠️ Make sure:")
        print("  1. Vector index 'law_docs_index' exists in MongoDB Atlas")
        print("  2. Documents have been ingested with 'embedding' field")
        print("  3. MongoDB connection is active")
        import traceback
        traceback.print_exc()
        return []
