import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

CHUNKS_FILE = Path("data/chunks.jsonl")
COLLECTION_NAME = "indian_laws"
VECTOR_DIM = 384  # all-MiniLM-L6-v2

def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    # Connect
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    
    # Check if collection exists, create if not
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
    
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print("Uploading points to Qdrant...")
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        payload = {
            "act_name": chunk["act_name"],
            "section_number": chunk["section_number"],
            "jurisdiction": chunk["jurisdiction"],
            "category": chunk["category"],
            "year": chunk["year"],
            "text": chunk["text"]
        }
        points.append(PointStruct(id=i, vector=vec.tolist(), payload=payload))
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"Upserted {min(i+batch_size, len(points))}/{len(points)}")
    
    print("Done! All chunks uploaded to Qdrant Cloud.")

if __name__ == "__main__":
    main()