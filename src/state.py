from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class AgentState(BaseModel):
    # --- Conversation History ---
    messages: list = []  # list of {"role": "user" | "assistant", "content": ...}

    # --- User Input ---
    user_query: str = ""
    document_path: Optional[str] = None
    document_text: Optional[str] = None

    # --- Analysis ---
    jurisdiction: Optional[Dict[str, str]] = None
    category: Optional[List[str]] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

    # --- Retrieval ---
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    web_search_results: Optional[str] = None

    # --- Document Analysis ---
    clause_analysis: Optional[List[Dict[str, Any]]] = None

    # --- Reasoning ---
    reasoning_chain: Optional[str] = None
    confidence_score: Optional[str] = None

    # --- Output ---
    action_output: Optional[str] = None
    escalation_needed: bool = False

    retry_count: int = 0          # tracks how many retrieval retries have been attempted
    retry_mode: bool = False      # if True, the Law Retriever will use an expanded search strategy

    user_choice: Optional[str] = None   # "generate_document" or "find_lawyer"