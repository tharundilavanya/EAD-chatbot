from db import get_collection, embed_text
import json

collection = get_collection()

print("=" * 50)
print("📊 MongoDB Collection Debug")
print("=" * 50)

# 1. Check total documents
doc_count = collection.count_documents({})
print(f"\n1️⃣ Total documents in collection: {doc_count}")

# 2. List first document to see structure
if doc_count > 0:
    first_doc = collection.find_one({})
    print(f"\n2️⃣ First document structure:")
    print(f"   Keys: {list(first_doc.keys())}")
    for key in first_doc.keys():
        if key == "_id":
            print(f"   - {key}: {first_doc[key]}")
        elif key == "embedding":
            embedding = first_doc[key]
            if isinstance(embedding, list):
                print(f"   - {key}: list with {len(embedding)} dimensions")
            else:
                print(f"   - {key}: {type(embedding)}")
        elif key == "text":
            text = first_doc[key]
            print(f"   - {key}: {text[:100]}..." if len(text) > 100 else f"   - {key}: {text}")
        else:
            print(f"   - {key}: {first_doc[key]}")

# 3. Check indexes
print(f"\n3️⃣ Available indexes:")
indexes = collection.list_indexes()
for idx in indexes:
    print(f"   - Name: {idx['name']}")
    print(f"     Keys: {idx['key']}")

# 4. Test embedding dimension
print(f"\n4️⃣ Testing embedding model:")
test_query = "test"
test_embedding = embed_text(test_query)
print(f"   Query: '{test_query}'")
print(f"   Embedding dimension: {len(test_embedding)}")
print(f"   First 5 values: {test_embedding[:5]}")

# 5. Check if embedding field is actually populated
embedding_count = collection.count_documents({"embedding": {"$exists": True}})
print(f"\n5️⃣ Documents with 'embedding' field: {embedding_count}")

text_count = collection.count_documents({"text": {"$exists": True}})
print(f"   Documents with 'text' field: {text_count}")

# 6. Try manual vector search
print(f"\n6️⃣ Attempting manual vector search:")
try:
    query = "what is the assignment about"
    query_vector = embed_text(query)
    
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_vector,
                "path": "embedding",
                "numCandidates": 50,
                "limit": 3,
                "index": "law_docs_index"
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    print(f"   Results found: {len(results)}")
    if results:
        for i, result in enumerate(results):
            print(f"   [{i+1}] Score: {result.get('score')}, Text: {result.get('text')[:100]}...")
    else:
        print("   ⚠️ No results returned")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
