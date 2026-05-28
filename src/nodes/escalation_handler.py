from state import AgentState

def escalation_handler(state: AgentState):
    
    reason = (
            "The case appears to involve disputes that are beyond the scope of this "
            "automated service. Only a licensed advocate can properly assess "
            "such situations."
        )

   
    checklist = ""

    # Add location
    if state.jurisdiction:

        state_name = state.jurisdiction.get("state", "?")

        country = state.jurisdiction.get(
            "country",
            "India"
        )

        checklist += (
            f"- 📍 Location: "
            f"{state_name}, {country}\n"
        )


    # Add legal category
    if state.category:

        checklist += (
            f"- ⚖️ Legal Area: "
            f"{', '.join(state.category)}\n"
        )


    # Add document info
    if state.document_text:

        checklist += (
            "- 📄 User uploaded a document. "
            "Bring a copy.\n"
        )


    # Add risky clause warning
    if state.clause_analysis:

        high_risk_count = 0

        for clause in state.clause_analysis:

            if clause.get("risk_level") == "high":

                high_risk_count += 1

        if high_risk_count > 0:

            checklist += (
                f"- 🚩 High-risk clauses found: "
                f"{high_risk_count}\n"
            )

    # General must‑bring items
    checklist += (
        "-  **Key documents to collect:**\n"
        "   • Any written agreement, contract, or notice you received.\n"
        "   • Proof of payments, receipts, bank statements.\n"
        "   • Identity proof (Aadhaar, PAN, etc.) and address proof.\n"
        "   • A chronological summary of events (dates, what happened).\n"
    )

    # ── 3. Compose the final message ──
    message = f"""
 **This case requires a licensed advocate.**

**Why?**
{reason}

---

### 📞 Get help now – free helplines (India‑wide)
- **NALSA Legal Aid Helpline:** Dial **15100** (24×7, free, multiple languages)
- **Tele‑Law (pre‑litigation advice):** Dial **14454** (free, connects you to a panel lawyer)
- **Nyaya Bandhu App:** Download on your phone to find pro bono advocates near you.

### 🏛️ Find a lawyer in your district
- **NALSA Directory of Legal Aid Clinics:** [https://nalsa.gov.in/legal-aid-clinics](https://nalsa.gov.in/legal-aid-clinics)  
  (Select your state → district to see the DLSA office address, phone, and email.)
- **Pro Bono Legal Services Portal:** [https://probono-doj.in](https://probono-doj.in)  
  (Connects you with volunteer lawyers across India.)

---

### 📋 What to share with your lawyer
{checklist}

### Important reminder
This automated system is a **first‑aid tool**, not a substitute for professional legal advice.  
Do **not** take any legal action based solely on this message. A qualified advocate will review your situation in full and guide you through the correct legal process.
"""

    # ── 4. Store in state ──
    state.action_output = message
    state.escalation_needed = True
    state.messages.append({"role": "assistant", "content": message})
    return state