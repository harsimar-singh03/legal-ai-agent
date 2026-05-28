import os
import json
from dotenv import load_dotenv
from groq import Groq
from state import AgentState
from models import CategoryOutput
from langgraph.types import interrupt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
VALID_CATEGORIES = ["consumer", "employment", "tenancy", "cyber", "RTI", "workplace_harassment", "other"]

def category_classifier(state: AgentState):
    # If we already have a category, skip this node
    if state.category:
        state.needs_clarification = False
        return state

    original_query = state.user_query

    while True:
        location_info = "Unknown"
        if state.jurisdiction:
            location_info = f"{state.jurisdiction.get('state', 'Unknown')}, {state.jurisdiction.get('country', 'Unknown')}"

        system_prompt = f"""
You are a legal category classifier for an Indian legal first‑aid system. Your job is to assign the user's problem to one or more of these legal categories: {', '.join(VALID_CATEGORIES)}.

IMPORTANT RULES:
- Only use "other" if the problem is clearly about criminal law (assault, theft, murder, fraud by an individual not related to a purchase), family law (divorce, custody), constitutional law, or any topic NOT covered by the other six categories.
- "cyber" is for online fraud, hacking, identity theft — not for physical violence committed by someone who happens to use a phone.
- "employment" is for salary disputes, wrongful termination, workplace benefits — not for a neighbor dispute.
- "workplace_harassment" is specifically for sexual harassment at work — not general harassment.
- "consumer" is for defective products, services, or unfair trade practices by a business.
- "RTI" is for seeking information from a public authority.
- "tenancy" is for landlord-tenant disputes over rent, deposit, eviction.

If the problem fits none of the first six categories, you MUST return ["other"] and set needs_clarification to false.

Return ONLY a JSON object with:
{{
  "categories": ["category1"],
  "needs_clarification": false,
  "clarification_question": null
}}
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        
        for msg in state.messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        result = CategoryOutput(**data)

        if not result.needs_clarification:
            state.category = result.categories
            state.needs_clarification = False
            state.clarification_question = None
            state.messages.append({"role": "assistant", "content": json.dumps(data)})
            return state
        else:
            new_info = interrupt(result.clarification_question)
            state.messages.append({"role": "assistant", "content": result.clarification_question})
            state.messages.append({"role": "user", "content": new_info})