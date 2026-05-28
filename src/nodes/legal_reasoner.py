import os
from dotenv import load_dotenv
from groq import Groq
from state import AgentState

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def legal_reasoner(state: AgentState):
    
    law_text = ""
    if state.retrieved_chunks:
        for i, chunk in enumerate(state.retrieved_chunks, 1):
            law_text += (
                f"Chunk {i}:\n"
                f"Act: {chunk.get('act_name', 'Unknown')}\n"
                f"Section: {chunk.get('section_number', 'Unknown')}\n"
                f"Text: {chunk.get('text', '')}\n\n"
            )
    else:
        law_text = "No specific law sections were retrieved. You must note this and recommend escalation to a lawyer."

    
    web_info = ""
    if state.web_search_results:
        web_info = f"Recent legal updates/web results:\n{state.web_search_results[:2000]}\n\n"

    
    doc_info = ""
    if state.clause_analysis:
        doc_info = "Document clause analysis (risks flagged):\n"
        for ca in state.clause_analysis:
            doc_info += (
                f"- Clause: {ca['clause_text'][:100]}...\n"
                f"  Risk: {ca['risk_level']}\n"
                f"  Reason: {ca['explanation']}\n"
                f"  Conflicting section: {ca.get('conflicting_section', 'N/A')}\n\n"
            )

    # prompt
    prompt = f"""
You are a legal reasoning engine for an Indian legal first-aid system. Your task is to analyse the user's situation using ONLY the legal information provided below. You must not invent any law or section that is not explicitly mentioned in the provided law chunks or web results.

### User's Problem
{state.user_query}

### Applicable Law (retrieved from official Bare Acts)
{law_text}

### {web_info}

### {doc_info if doc_info else ""}

### Instructions
Follow this exact structure in your response:

1. **Applicable Law and Section(s)**
   - Identify the most relevant act and specific section numbers that apply to the user's problem. You must cite the exact section number(s) from the chunks above.

2. **User's Legal Rights**
   - Explain what rights the user has under those sections.

3. **Possible Violation**
   - State whether the user's description indicates a violation of those rights. Be clear and specific.

4. **Recommended Next Steps**
   - Provide a concrete, step-by-step action plan (e.g., send a legal notice, file a complaint with a specific authority, approach the Consumer Forum, etc.). Mention any time limits if stated in the law.

5. **Important Caveats**
   - Note any missing information that would be needed for a definitive opinion, and if the case seems too complex, recommend consulting a lawyer.

Remember: if the provided legal information does not support a definitive answer, you must clearly state that and recommend escalation to a licensed advocate.
"""
    messages = [
        {"role": "system", "content": "You are a precise Indian legal reasoning assistant. Every legal claim must be backed by a section number from the provided context."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )

    reasoning = response.choices[0].message.content
    state.reasoning_chain = reasoning
    state.messages.append({"role": "assistant", "content": reasoning})
    return state