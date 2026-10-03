import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
from groq import Groq
from langgraph.types import Command
from pydantic import BaseModel

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.graph import app as langgraph_app
from src.state import AgentState

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

api = FastAPI(
    title="Indian Legal First-Aid Agent API",
    description=(
        "Production REST API for the 9-node LangGraph Indian Legal First-Aid Agent. "
        "Supports hybrid Qdrant vector retrieval, Tavily Supreme Court case search, "
        "contract risk auditing, and automated legal notice generation."
    ),
    version="2.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for frontend assets
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory session store keyed by thread_id (pairs with SQLite checkpointer)
SESSIONS: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(thread_id: str) -> Dict[str, Any]:
    if thread_id not in SESSIONS:
        SESSIONS[thread_id] = {
            "thread_id": thread_id,
            "state": AgentState(),
            "pdf_path": None,
            "pdf_filename": None,
            "waiting_for_input": False,
            "pending_question": None,
            "graph_finished": False,
            "last_reasoning_shown": None,
            "last_clauses_shown": None,
        }
    return SESSIONS[thread_id]


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    answer: str


class ResetRequest(BaseModel):
    thread_id: Optional[str] = None


class AgentResponse(BaseModel):
    thread_id: str
    replies: List[str]
    waiting_for_input: bool
    pending_question: Optional[str] = None
    is_choice_prompt: bool = False
    graph_finished: bool
    has_pdf_download: bool
    jurisdiction: Optional[Dict[str, Optional[str]]] = None
    category: Optional[List[str]] = None
    clause_analysis: Optional[List[Dict[str, Any]]] = None
    escalation_needed: bool = False


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def format_clause_analysis(clauses: List[Dict[str, Any]]) -> str:
    analysis = "### 🔍 Document Clause Analysis\n\n"
    emoji_map = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    for clause in clauses:
        emoji = emoji_map.get(clause.get("risk_level"), "⚪")
        risk_badge = str(clause.get("risk_level", "unknown")).upper()
        section_ref = clause.get("conflicting_section")
        conflict_str = f" *(Conflicts with: {section_ref})*" if section_ref else ""
        analysis += (
            f"{emoji} **[{risk_badge} RISK]** `{clause['clause_text'][:100]}...`{conflict_str}\n"
            f"- {clause['explanation']}\n\n"
        )
    return analysis


def generate_post_graph_reply(user_text: str, state: AgentState) -> str:
    summary = ""
    if state.user_query:
        summary += f"Problem: {state.user_query}\n"
    if state.category:
        summary += f"Legal Area: {', '.join(state.category)}\n"
    if state.reasoning_chain:
        summary += f"Reasoning: {state.reasoning_chain[:500]}\n"

    system_prompt = f"""
You are a helpful Indian legal assistant.

Case Summary:
{summary}

User message:
"{user_text}"

Answer briefly and helpfully.
Stay within Indian law.

Language Rule: If the user's message is written in Hindi or Hinglish (transliterated Hindi), you must reply in Hindi (Devanagari script). Otherwise, reply in English.
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,
        max_tokens=2048,
        reasoning_format="hidden",
    )
    return response.choices[0].message.content


def execute_graph_turn(
    session: Dict[str, Any],
    user_input: Optional[str] = None,
    resume_answer: Optional[str] = None,
) -> AgentResponse:
    thread_id = session["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}
    state: AgentState = session["state"]
    replies: List[str] = []

    if user_input:
        state.user_query = user_input
        if not state.messages or state.messages[-1].get("content") != user_input:
            state.messages.append({"role": "user", "content": user_input})

    if session["pdf_path"] and not state.document_path:
        state.document_path = session["pdf_path"]

    if resume_answer is not None:
        result = langgraph_app.invoke(Command(resume=resume_answer), config)
    else:
        result = langgraph_app.invoke(state, config)

    # Update session state
    updated_state = AgentState(**result)
    session["state"] = updated_state

    # Check if new legal reasoning was produced
    reasoning = result.get("reasoning_chain")
    if reasoning and reasoning != session["last_reasoning_shown"]:
        replies.append(f"### 📝 Legal Reasoning\n\n{reasoning}")
        session["last_reasoning_shown"] = reasoning

    # Check if new document clause analysis was produced
    clauses = result.get("clause_analysis")
    if clauses and clauses != session["last_clauses_shown"]:
        replies.append(format_clause_analysis(clauses))
        session["last_clauses_shown"] = clauses

    # Check for Human-in-the-Loop interrupts
    interrupts = result.get("__interrupt__")
    if interrupts:
        question = interrupts[0].value
        session["pending_question"] = question
        session["waiting_for_input"] = True
        session["graph_finished"] = False
        replies.append(question)
        is_choice = "Based on the legal analysis, I can either:" in str(question)
        return AgentResponse(
            thread_id=thread_id,
            replies=replies,
            waiting_for_input=True,
            pending_question=question,
            is_choice_prompt=is_choice,
            graph_finished=False,
            has_pdf_download=bool(updated_state.action_output),
            jurisdiction=updated_state.jurisdiction,
            category=updated_state.category,
            clause_analysis=updated_state.clause_analysis,
            escalation_needed=updated_state.escalation_needed,
        )

    # Graph completed
    session["waiting_for_input"] = False
    session["pending_question"] = None
    session["graph_finished"] = True

    action_output = result.get("action_output")
    if action_output:
        replies.append(action_output)

    return AgentResponse(
        thread_id=thread_id,
        replies=replies,
        waiting_for_input=False,
        pending_question=None,
        is_choice_prompt=False,
        graph_finished=True,
        has_pdf_download=bool(updated_state.action_output),
        jurisdiction=updated_state.jurisdiction,
        category=updated_state.category,
        clause_analysis=updated_state.clause_analysis,
        escalation_needed=updated_state.escalation_needed,
    )


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────
@api.get("/", include_in_schema=False)
async def serve_ui():
    """Serves the production Legal-Tech Web UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return Response("Frontend index.html not found.", status_code=404)
    return FileResponse(str(index_file))


@api.post("/api/reset")
async def reset_session(req: ResetRequest):
    """Creates a clean thread session for a new legal case."""
    new_thread_id = f"case_{int(time.time() * 1000)}"
    get_or_create_session(new_thread_id)
    return {"thread_id": new_thread_id, "status": "reset"}


@api.post("/api/upload")
async def upload_pdf(
    thread_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Uploads a PDF agreement/contract and attaches it to the active case session."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session = get_or_create_session(thread_id)
    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        session["pdf_path"] = tmp.name
        session["pdf_filename"] = file.filename

    return {
        "status": "uploaded",
        "thread_id": thread_id,
        "filename": file.filename,
    }


@api.post("/api/chat", response_model=AgentResponse)
async def chat_endpoint(req: ChatRequest):
    """Starts the LangGraph legal pipeline or answers follow-up questions after completion."""
    session = get_or_create_session(req.thread_id)

    # If the graph is currently waiting on an interrupt, route seamlessly to resume
    if session["waiting_for_input"]:
        return execute_graph_turn(session, resume_answer=req.message)

    # If the graph already finished, handle conversational follow-up
    if session["graph_finished"]:
        state: AgentState = session["state"]
        reply = generate_post_graph_reply(req.message, state)
        return AgentResponse(
            thread_id=req.thread_id,
            replies=[reply],
            waiting_for_input=False,
            pending_question=None,
            is_choice_prompt=False,
            graph_finished=True,
            has_pdf_download=bool(state.action_output),
            jurisdiction=state.jurisdiction,
            category=state.category,
            clause_analysis=state.clause_analysis,
            escalation_needed=state.escalation_needed,
        )

    # Initial graph invocation
    return execute_graph_turn(session, user_input=req.message)


@api.post("/api/resume", response_model=AgentResponse)
async def resume_endpoint(req: ResumeRequest):
    """Resumes a paused Human-in-the-Loop LangGraph execution."""
    session = get_or_create_session(req.thread_id)
    return execute_graph_turn(session, resume_answer=req.answer)


@api.get("/api/download-pdf/{thread_id}")
async def download_pdf(thread_id: str):
    """Compiles and streams the generated legal document as a downloadable PDF."""
    session = SESSIONS.get(thread_id)
    if not session or not session["state"].action_output:
        raise HTTPException(status_code=404, detail="No generated document available for this session.")

    raw_text = session["state"].action_output
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    clean_text = raw_text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 7, clean_text)

    pdf_bytes = bytes(pdf.output())
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="legal_document_{thread_id}.pdf"'
        },
    )
