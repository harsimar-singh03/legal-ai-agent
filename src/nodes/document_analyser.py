import os
import json
from dotenv import load_dotenv
from groq import Groq
from state import AgentState
from models import DocumentAnalysisOutput

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_clauses(text):
    # Split document by paragraph
    clauses = text.split("\n\n")
   
    clean_clauses = []
    for clause in clauses:
        clause = clause.strip()
        if clause:
            clean_clauses.append(clause)
    return clean_clauses

def document_analyser(state: AgentState):
    print("===== DOCUMENT ANALYSER NODE EXECUTED =====")
    if not state.document_text:
        return state  

    # Extract clauses
    clauses = extract_clauses(state.document_text)
    if not clauses:
        return state

    # Build law context from retrieved chunks 
    law_context = ""
    if state.retrieved_chunks:
        law_parts = []
        for chunk in state.retrieved_chunks[:5]:
            text = (
                f"{chunk['act_name']} "
                f"Section {chunk['section_number']} : "
                f"{chunk['text'][:300]}"
            )
            law_parts.append(text)

        law_context = "\n\n".join(law_parts)

   
    prompt = f"""
You are a legal contract reviewer. Review the following clauses from a document against the provided legal context (if any). For each clause, determine:

- Does it conflict with any legal provision in the context?
- Is it unusually unfair, one‑sided, or risky for the person signing it?
- Assign a risk level: low, medium, high.
- Provide a short explanation and cite the relevant section if a conflict exists.

Legal context (from retrieved law):
{law_context if law_context else "None available – base your review on general Indian contract fairness principles."}

Clauses to review :
{json.dumps(clauses, indent=2)}

Return a JSON object with:
- flagged_clauses: a list of objects, each with: clause_text, risk_level, explanation, conflicting_section (or null if no specific section).
- summary: a 2 to 3 sentence overall assessment of whether it's safe to sign this document.
"""
    messages = []
    # Include conversation history (optional for context)
    for msg in state.messages[-4:]:   # only last few messages to keep prompt small
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    print("DEBUG document analyser raw output:", raw[:500])   

    data = json.loads(raw)
    result = DocumentAnalysisOutput(**data)

     # Save clause analysis
    state.clause_analysis = []
    for clause in result.flagged_clauses:
        state.clause_analysis.append(
            clause.dict()
        )
    # Save summary
    state.action_output = result.summary
    
    
    state.messages.append({
        "role": "assistant",
        "content": raw
    })
    return state