import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

COLLECTION_NAME = "indian_laws"
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, category=None, jurisdiction=None, top_k=5):
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    query_vec = model.encode([query])[0].tolist()
    
    must_conditions = []
    if category:
        must_conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if jurisdiction:
        must_conditions.append(FieldCondition(key="jurisdiction", match=MatchValue(value=jurisdiction)))
    
    query_filter = Filter(must=must_conditions) if must_conditions else None
    
    # Use correct method based on qdrant-client version
    try:
        # Newer versions (1.7+)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            query_filter=query_filter
        ).points
    except AttributeError:
        # Older versions
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter
        )
    return results

if __name__ == "__main__":
    q = "My landlord is not returning my security deposit in Mumbai"
    print(f"Query: {q}\n")
    hits = search(q, category="tenancy", jurisdiction="Maharashtra")
    for i, hit in enumerate(hits):
        score = hit.score
        payload = hit.payload
        print(f"--- Result {i+1} (score: {score:.4f}) ---")
        print(f"Act: {payload['act_name']}")
        print(f"Section: {payload['section_number']}")
        print(f"Text: {payload['text'][:200]}...\n")