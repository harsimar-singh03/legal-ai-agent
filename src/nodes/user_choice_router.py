import json
import os
from dotenv import load_dotenv
from groq import Groq
from state import AgentState
from langgraph.types import interrupt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def user_choice_router(state: AgentState):
    """
    ask the user whether they want
    a ready‑to‑send document or a lawyer referral.
    """
    question = (
        "Based on the legal analysis, I can either:\n"
        "1. Generate a ready‑to‑send legal document (complaint letter / legal notice) for you, OR\n"
        "2. Provide information to help you find a licensed advocate.\n\n"
        "Which would you prefer? Please type your choice in your own words."
    )

    user_response = interrupt(question)
    state.messages.append({"role": "assistant", "content": question})
    state.messages.append({"role": "user", "content": user_response})

    # ── Use LLM to interpret the user's intent from free‑form text ──
    prompt = f"""
A user was asked whether they want a ready‑to‑send legal document or a lawyer referral.
Their exact response was:

"{user_response}"

Classify their intent as one of:
- "generate_document" → they want a legal notice / complaint letter generated
- "find_lawyer" → they want help finding a licensed advocate

Return ONLY a JSON object:
{{
  "intent": "generate_document" or "find_lawyer"
}}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an intent classifier. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    intent = data.get("intent", "find_lawyer")

    # Store the choice in state so we can route
    state.user_choice = intent
    return state