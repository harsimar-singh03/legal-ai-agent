import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from tavily import TavilyClient
from state import AgentState

load_dotenv()


import json
from pathlib import Path
import numpy as np

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=4,
    check_compatibility=False,
)
model = SentenceTransformer("all-MiniLM-L6-v2")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

COLLECTION_NAME = "indian_laws"
CHUNKS_PATH = Path(__file__).parent.parent.parent / "data" / "chunks.jsonl"
_LOCAL_CHUNKS_CACHE = None


def get_local_chunks():
    global _LOCAL_CHUNKS_CACHE
    if _LOCAL_CHUNKS_CACHE is None and CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            _LOCAL_CHUNKS_CACHE = [json.loads(line) for line in f if line.strip()]
    return _LOCAL_CHUNKS_CACHE or []


def fallback_local_search(query_vec, state: AgentState, top_k=8):
    """High-availability local fallback over data/chunks.jsonl if Qdrant Cloud cluster is hibernated."""
    chunks = get_local_chunks()
    if not chunks:
        return []

    allowed_jurisdictions = {"India"}
    if state.jurisdiction and state.jurisdiction.get("state"):
        allowed_jurisdictions.add(state.jurisdiction["state"])

    allowed_categories = set(state.category) if state.category else set()

    filtered = [
        c for c in chunks
        if (c.get("jurisdiction") in allowed_jurisdictions)
        and (not allowed_categories or c.get("category") in allowed_categories)
    ]
    if not filtered:
        filtered = chunks[:100]

    # Pre-filter by keyword overlap if filtered pool is large to keep latency <200ms
    query_words = set(state.user_query.lower().split())
    if len(filtered) > 60 and query_words:
        filtered.sort(
            key=lambda c: sum(1 for w in query_words if w in c.get("text", "").lower()),
            reverse=True,
        )
        filtered = filtered[:60]

    texts = [c.get("text", "") for c in filtered]
    doc_vecs = model.encode(texts, normalize_embeddings=True)
    q_vec = np.array(query_vec)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    scores = np.dot(doc_vecs, q_norm)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        c = filtered[int(idx)]
        results.append({
            "act_name": c.get("act_name"),
            "section_number": c.get("section_number"),
            "text": c.get("text"),
            "score": float(scores[int(idx)]),
        })
    return results


def search_qdrant(query_vec, query_filter, top_k=8):
    """Version-agnostic search for Qdrant."""
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            query_filter=query_filter
        ).points
    except AttributeError:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter
        )
    return results

def law_retriever(state: AgentState):
    
    if state.retrieved_chunks and not state.retry_mode:
        return state

   
    query_text = state.user_query
    if state.document_text:
        query_text += " " + state.document_text[:1000]

    
    query_filter = None
    if not state.retry_mode:
        must_conditions = []

       
        if state.jurisdiction and state.jurisdiction.get("state"):
            must_conditions.append(
                FieldCondition(
                    key="jurisdiction",
                    match=MatchAny(any=[state.jurisdiction["state"], "India"])
                )
            )

        # Category filter
        if state.category:
            must_conditions.append(
                FieldCondition(key="category", match=MatchAny(any=state.category))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

    
    top_k = 10 if state.retry_mode else 8

   
    query_vec = model.encode([query_text])[0].tolist()
    retrieved_chunks = []
    try:
        hits = search_qdrant(query_vec, query_filter, top_k=top_k)
        for hit in hits:
            retrieved_chunks.append({
                "act_name": hit.payload.get("act_name"),
                "section_number": hit.payload.get("section_number"),
                "text": hit.payload.get("text"),
                "score": hit.score
            })
    except Exception as e:
        print(f"[law_retriever] Qdrant Cloud unreachable ({e}), using local chunks.jsonl fallback.")
        retrieved_chunks = fallback_local_search(query_vec, state, top_k=top_k)

    # 6. Tavily web search 
    web_results_text = ""
    try:
        location = state.jurisdiction.get("state", "") if state.jurisdiction else ""
        tavily_query = f"recent Supreme Court judgment {state.user_query} {location}"
        max_results = 5 if state.retry_mode else 3
        tavily_response = tavily.search(query=tavily_query, search_depth="basic", max_results=max_results)
        snippets = [r.get("content", "") for r in tavily_response.get("results", [])]
        web_results_text = "\n\n".join(snippets)
    except Exception as e:
        web_results_text = f"[Web search error: {e}]"

    # Store results
    state.retrieved_chunks = retrieved_chunks
    state.web_search_results = web_results_text

    
    if state.retry_mode:
        state.retry_mode = False

    return state