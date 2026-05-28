import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from tavily import TavilyClient
from state import AgentState

load_dotenv()


qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

COLLECTION_NAME = "indian_laws"

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
    hits = search_qdrant(query_vec, query_filter, top_k=top_k)

    
    retrieved_chunks = []
    for hit in hits:
        retrieved_chunks.append({
            "act_name": hit.payload.get("act_name"),
            "section_number": hit.payload.get("section_number"),
            "text": hit.payload.get("text"),
            "score": hit.score
        })

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