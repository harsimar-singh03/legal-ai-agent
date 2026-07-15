import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from state import AgentState
from langgraph.types import interrupt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def find_next_placeholder(text):
    if not text:
        return None
    # Find all pattern [Placeholder] but ignore headers
    for match in re.finditer(r"\[(?!LEGAL_NOTICE|COMPLAINT_LETTER|ACTION_PLAN)(.*?)\]", text):
        return match.group(1)
    return None

def action_generator(state: AgentState):
    # If the draft template is not yet generated, build it.
    if not state.action_output or not state.action_output.startswith(("[LEGAL_NOTICE]", "[COMPLAINT_LETTER]", "[ACTION_PLAN]")):
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
- MINIMIZE PLACEHOLDERS: Extract any dates, amounts, names, addresses, and details already provided in the conversation history and populate them directly. Do NOT create placeholders for them.
- Keep placeholders to a strict minimum (maximum 4-5 total). Do not generate redundant placeholders (e.g. do not ask for multiple different dates like vacate date, lease start date, lease end date, and termination date—just use a single lease date and vacation date).
- If any personal details are completely unknown, use descriptive placeholders in square brackets, e.g., `[Your Full Name]`, `[Landlord's Name]`, `[Landlord's Address]`.
- ONLY ask for information that is clearly required and missing.

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
            {"role": "system", "content": "You are a precise legal document assistant. Output a flat JSON object. 'content' is always a single string, never an object. Always generate the document content in English, even if the user's initial query or conversation history is in Hindi or Hinglish."}
        ]
        # Include conversation history so the LLM can extract details the user has already provided in chat
        for msg in state.messages:
            if isinstance(msg, dict) and "content" in msg:
                # Skip raw JSON nodes output from helper nodes
                if not msg["content"].strip().startswith("{"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            reasoning_format="hidden"
        )

        raw = response.choices[0].message.content
        print(raw)
        data = json.loads(raw)
        output_type = data.get("output_type", "legal_notice")
        content = data.get("content", "")
        # Clean escaped newline characters to actual newlines
        content = content.replace("\\n", "\n")

        state.action_output = f"[{output_type.upper()}]\n\n{content}"

    # Find the next placeholder to fill
    placeholder = find_next_placeholder(state.action_output)
    if placeholder:
        question = f"Please provide: {placeholder}"
        
        # This will pause graph execution and yield control to user
        answer = interrupt(question)
        
        # Save messages to conversation history
        state.messages.append({
            "role": "assistant",
            "content": question
        })
        state.messages.append({
            "role": "user",
            "content": answer
        })
        
        # Substitute the placeholder in action_output
        state.action_output = state.action_output.replace(
            f"[{placeholder}]",
            answer
        )
    else:
        # No more placeholders! Append final output to state messages.
        if not state.messages or state.messages[-1].get("content") != state.action_output:
            state.messages.append({
                "role": "assistant",
                "content": state.action_output
            })

    return state