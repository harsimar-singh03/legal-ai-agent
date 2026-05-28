from state import AgentState

def confidence_evaluator(state: AgentState):
    # Compute average retrieval score (purely informational)
    chunks = state.retrieved_chunks or []
    if chunks:
        scores = [c.get("score", 0) for c in chunks if "score" in c]
        avg_score = sum(scores) / len(scores) if scores else 0.0
    else:
        avg_score = 0.0

    
    if avg_score >= 0.6:
        confidence = "high"
    elif avg_score >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    
    reasoning = state.reasoning_chain or ""
    if any(phrase in reasoning.lower() for phrase in ["contradict", "conflicting", "not clear", "insufficient"]):
        if confidence == "high":
            confidence = "medium"
        elif confidence == "medium":
            confidence = "low"

    state.confidence_score = confidence
   
    state.retry_mode = False
    state.retry_count = 0
    return state