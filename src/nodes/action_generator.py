import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from state import AgentState
from langgraph.types import interrupt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def action_generator(state: AgentState):
    reasoning = state.reasoning_chain or "No legal reasoning available."
    user_query = state.user_query
    jurisdiction = state.jurisdiction or {}
    category = state.category or []

   
    prompt = f"""
You are a legal document generator. Generate a formal document based on the provided legal reasoning. Choose the most appropriate type: legal notice, complaint letter, or action plan.

CRITICAL RULES:
- Output must be a JSON object with two fields: "output_type" (string) and "content" (string).
- The "content" field MUST be a single plain string containing the entire document, with line breaks as \\n.
- Do NOT nest any objects inside "content". It must be a string.
- If any personal details are unknown, use descriptive placeholders in square brackets, e.g., `[Your Full Name]`, `[Landlord's Name]`, `[Landlord's Address]`.
- ONLY ask for information that is clearly required.
- DO NOT invent multiple incidents/events unless explicitly mentioned by the user.
- If the user mentioned only one incident, generate placeholders for only one incident.

Important Rules:
- ONLY create placeholders for information actually needed.
- DO NOT assume multiple incidents.
- If the user described only one event, ask only for one event.
- Keep the document simple and realistic.

### User's Situation
{user_query}

### Jurisdiction & Category
- Jurisdiction: {jurisdiction.get('state', 'Unknown')}, {jurisdiction.get('country', 'India')}
- Category: {', '.join(category) if category else 'Not specified'}

### Legal Reasoning
{reasoning}

Return ONLY a JSON object like:
{{
  "output_type": "legal_notice",
  "content": "To [Landlord's Name],\\n[Landlord's Address]\\nMumbai\\n\\nSubject: ...\\n\\nSincerely,\\n[Your Full Name]"
}}
"""
    messages = [
        {"role": "system", "content": "You are a precise legal document assistant. Output a flat JSON object. 'content' is always a single string, never an object."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    print(raw)
    data = json.loads(raw)
    output_type = data.get("output_type", "legal_notice")
    content = data.get("content", "")
    

    
    placeholders = re.findall(
        r"\[(.*?)\]",
        content
    )

    
    unique_placeholders = []

    for ph in placeholders:

        if ph not in unique_placeholders:

            unique_placeholders.append(ph)


   
    answers = {}


    
    for placeholder in unique_placeholders:

        question = f"Please provide: {placeholder}"

        answer = interrupt(question)

        # Save messages
        state.messages.append({
            "role": "assistant",
            "content": question
        })

        state.messages.append({
            "role": "user",
            "content": answer
        })

        # Save answer
        answers[placeholder] = answer


    # Replace placeholders AFTER collecting all answers
    for placeholder, answer in answers.items():

        content = content.replace(
            f"[{placeholder}]",
            answer
        )


    # Final formatted output
    final_output = (
        f"[{output_type.upper()}]\n\n"
        f"{content}"
    )


    # Save final output
    state.action_output = final_output


    # Save in conversation history
    state.messages.append({
        "role": "assistant",
        "content": final_output
    })


    return state