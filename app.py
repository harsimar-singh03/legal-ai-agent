import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
import time
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from fpdf import FPDF

from src.state import AgentState
from src.graph import app
from langgraph.types import Command


# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
try:
    import streamlit as st
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        for key in ["GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "TAVILY_API_KEY",
                     "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_API_KEY"]:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
except Exception:
    pass  # not on Streamlit Cloud, .env will be loaded by dotenv

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ─────────────────────────────────────────────
# Streamlit page setup
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Legal First-Aid Agent",
    layout="wide"
)

st.title("⚖️ Indian Legal First-Aid Agent")


# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
defaults = {
    "thread_id": f"user_{int(time.time())}",
    "state": AgentState(),
    "chat_history": [],
    "waiting_for_input": False,
    "pdf_path": None,
    "last_reasoning_shown": None,
    "last_clauses_shown": None,
    "pending_question": None,
    "graph_finished": False,
    "start_processing": False,
    "pending_user_input": None,
    # ── NEW: deferred answer slots so the user message
    #         always appears on screen before processing begins
    "pending_resume_answer": None,
    "pending_followup": None,
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def add_message(role, content):

    st.session_state.chat_history.append(
        (role, content)
    )


def reset_session():

    st.session_state.state = AgentState()

    st.session_state.chat_history = []

    st.session_state.waiting_for_input = False

    st.session_state.pdf_path = None

    st.session_state.last_reasoning_shown = None

    st.session_state.last_clauses_shown = None

    st.session_state.pending_question = None

    st.session_state.graph_finished = False

    st.session_state.start_processing = False

    st.session_state.pending_user_input = None

    st.session_state.pending_resume_answer = None

    st.session_state.pending_followup = None

    st.session_state.thread_id = (
        f"user_{int(time.time())}"
    )


def show_reasoning(result):

    reasoning = result.get(
        "reasoning_chain"
    )

    if not reasoning:
        return

    if reasoning == st.session_state.last_reasoning_shown:
        return

    add_message(
        "assistant",
        f"📝 Legal Reasoning\n\n{reasoning}"
    )

    st.session_state.last_reasoning_shown = (
        reasoning
    )


def show_clause_analysis(result):

    clauses = result.get(
        "clause_analysis"
    )

    if not clauses:
        return

    if clauses == st.session_state.last_clauses_shown:
        return

    analysis = (
        "🔍 Document Clause Analysis\n\n"
    )

    emoji_map = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴"
    }

    for clause in clauses:

        emoji = emoji_map.get(
            clause.get("risk_level"),
            "⚪"
        )

        analysis += (
            f"{emoji} "
            f"{clause['clause_text'][:80]}...\n"
            f"{clause['explanation']}\n\n"
        )

    add_message(
        "assistant",
        analysis
    )

    st.session_state.last_clauses_shown = (
        clauses
    )


def run_agent(
    user_input=None,
    resume_answer=None
):

    config = {
        "configurable": {
            "thread_id": (
                st.session_state.thread_id
            )
        }
    }

    state = st.session_state.state

    # Save initial query
    if user_input:

        state.user_query = user_input

        if (
            not state.messages
            or state.messages[-1].get(
                "content"
            ) != user_input
        ):

            state.messages.append({
                "role": "user",
                "content": user_input
            })

    # Attach uploaded PDF
    if (
        st.session_state.pdf_path
        and not state.document_path
    ):

        state.document_path = (
            st.session_state.pdf_path
        )

    # Run graph
    if resume_answer is not None:

        result = app.invoke(
            Command(
                resume=resume_answer
            ),
            config
        )

    else:

        result = app.invoke(
            state,
            config
        )

    # Save updated state
    st.session_state.state = AgentState(
        **result
    )

    # Show reasoning
    show_reasoning(result)

    # Show clause analysis
    show_clause_analysis(result)

    # Handle interrupts
    interrupts = result.get(
        "__interrupt__"
    )

    if interrupts:

        question = (
            interrupts[0].value
        )

        st.session_state.pending_question = (
            question
        )

        add_message(
            "assistant",
            question
        )

        st.session_state.waiting_for_input = True

        st.session_state.graph_finished = False

        return

    # Graph completed
    st.session_state.waiting_for_input = False

    st.session_state.pending_question = None

    action_output = result.get(
        "action_output"
    )

    if action_output:

        add_message(
            "assistant",
            action_output
        )

    st.session_state.graph_finished = True


def generate_reply(
    system_prompt,
    user_text
):

    response = (
        groq_client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],

            temperature=0.7,

            max_tokens=400
        )
    )

    return (
        response
        .choices[0]
        .message.content
    )


def generate_post_graph_reply(
    user_text,
    state
):

    summary = ""

    if state.user_query:

        summary += (
            f"Problem: "
            f"{state.user_query}\n"
        )

    if state.category:

        summary += (
            f"Legal Area: "
            f"{', '.join(state.category)}\n"
        )

    if state.reasoning_chain:

        summary += (
            f"Reasoning: "
            f"{state.reasoning_chain[:500]}\n"
        )

    system_prompt = f"""
You are a helpful Indian legal assistant.

Case Summary:
{summary}

User message:
"{user_text}"

Answer briefly and helpfully.

Stay within Indian law.
"""

    return generate_reply(
        system_prompt,
        user_text
    )


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:

    st.header("📄 Your Case")

    user_text = st.text_area(
        "Describe your legal problem",
        height=200
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    # Save uploaded PDF
    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            st.session_state.pdf_path = (
                tmp.name
            )

        st.success(
            "PDF uploaded."
        )

    submit = st.button(
        "Submit",
        type="primary"
    )

    # Reset button
    if st.button("🆕 New Case"):

        reset_session()

        st.rerun()


# ─────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────
st.header("💬 Conversation")

for role, msg in (
    st.session_state.chat_history
):

    with st.chat_message(role):

        st.markdown(msg)


# ─────────────────────────────────────────────
# Run backend processing AFTER UI refresh
# ─────────────────────────────────────────────

# ── Deferred: initial submit ──────────────────
if st.session_state.start_processing:

    st.session_state.start_processing = False

    user_input = (
        st.session_state.pending_user_input
    )

    with st.spinner("🧠 Thinking..."):

        run_agent(
            user_input=user_input
        )

    st.rerun()


# ── Deferred: clarification answer ───────────
# The user message was already added to chat_history and st.rerun()
# was called, so the message is visible before we hit this block.
if st.session_state.pending_resume_answer is not None:

    resume_answer = (
        st.session_state.pending_resume_answer
    )

    st.session_state.pending_resume_answer = None

    with st.spinner("🧠 Thinking..."):

        run_agent(
            resume_answer=resume_answer
        )

    st.rerun()


# ── Deferred: post-graph follow-up ────────────
# Same pattern — message shown first, then we process here.
if st.session_state.pending_followup is not None:

    followup = (
        st.session_state.pending_followup
    )

    st.session_state.pending_followup = None

    with st.spinner("🧠 Thinking..."):

        bot_reply = generate_post_graph_reply(
            followup,
            st.session_state.state
        )

    add_message(
        "assistant",
        bot_reply
    )

    st.rerun()


# ─────────────────────────────────────────────
# Initial submission
# ─────────────────────────────────────────────
if submit and user_text:

    # Show user message instantly
    add_message(
        "user",
        user_text
    )

    # Save input for later processing
    st.session_state.pending_user_input = (
        user_text
    )

    # Trigger backend processing
    st.session_state.start_processing = True

    st.rerun()


# ─────────────────────────────────────────────
# Clarification handling
# ─────────────────────────────────────────────
if st.session_state.waiting_for_input:

    user_reply = st.chat_input(
        "Your answer..."
    )

    if user_reply:

        add_message(
            "user",
            user_reply
        )

        # Store the answer; deferred block above will pick it up
        # on the NEXT rerun — after the message is already on screen.
        st.session_state.pending_resume_answer = (
            user_reply
        )

        st.rerun()


# ─────────────────────────────────────────────
# Post-graph conversation
# ─────────────────────────────────────────────
elif st.session_state.graph_finished:

    user_reply = st.chat_input(
        "Ask a follow-up question..."
    )

    if user_reply:

        add_message(
            "user",
            user_reply
        )

        # Store the question; deferred block above will pick it up
        # on the NEXT rerun — after the message is already on screen.
        st.session_state.pending_followup = (
            user_reply
        )

        st.rerun()


# ─────────────────────────────────────────────
# PDF Download
# ─────────────────────────────────────────────
if st.session_state.state.action_output:

    st.divider()

    # Create PDF
    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        size=12
    )

    text = (
        st.session_state.state.action_output
        .encode("latin-1", "replace")
        .decode("latin-1")
    )

    pdf.multi_cell(
        0,
        10,
        text
    )

    # Convert PDF to bytes
    pdf_bytes = bytes(
        pdf.output(dest="S")
    )

    # Single download button
    st.download_button(
        label="📥 Download PDF",
        data=pdf_bytes,
        file_name="legal_document.pdf",
        mime="application/pdf"
    )