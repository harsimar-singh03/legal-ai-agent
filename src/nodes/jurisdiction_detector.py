import os
import json
from dotenv import load_dotenv
from groq import Groq
from state import AgentState
from models import JurisdictionOutput
from langgraph.types import interrupt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def jurisdiction_detector(state: AgentState):
    # If we already know the jurisdiction, skip this node
    if state.jurisdiction and state.jurisdiction.get("state"):
        state.needs_clarification = False
        return state

    # The original user problem is stored in state.user_query.
    original_query = state.user_query

    while True:
        # Build system prompt FIRST
        system_prompt = f"""
You are a legal assistant for Indian citizens. Your task is to determine the country and state from the user's description.
The user initially described a problem, and may have provided their location in a follow‑up message.

Original problem: "{original_query}"

Rules:
- If the conversation now contains a city or state (e.g. "Mumbai"), infer the state.
- If no location is mentioned anywhere in the conversation, ask a short question.

Return ONLY a JSON object:
{{
  "country": "India",
  "state": "Maharashtra",   // or null
  "needs_clarification": false,
  "clarification_question": null
}}
"""
        # Place system prompt at index 0
        messages = [{"role": "system", "content": system_prompt}]
        
        # Then append conversation history
        for msg in state.messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        result = JurisdictionOutput(**data)

        if not result.needs_clarification:
            state.jurisdiction = {"country": result.country, "state": result.state}
            state.needs_clarification = False
            state.clarification_question = None
            state.messages.append({"role": "assistant", "content": json.dumps(data)})
            return state
        else:
            # Need clarification – pause the graph
            new_info = interrupt(result.clarification_question)
            state.messages.append({"role": "assistant", "content": result.clarification_question})
            state.messages.append({"role": "user", "content": new_info})