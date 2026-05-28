from pydantic import BaseModel
from typing import Optional

class JurisdictionOutput(BaseModel):
    country: str
    state: Optional[str] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

class CategoryOutput(BaseModel):
    categories: list[str]              # one or more of the six predefined categories
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

class ClauseRisk(BaseModel):
    clause_text: str
    risk_level: str          # "low", "medium", "high"
    explanation: str
    conflicting_section: Optional[str] = None   # e.g., "Section 35 of Consumer Protection Act"

class DocumentAnalysisOutput(BaseModel):
    flagged_clauses: list[ClauseRisk]
    summary: str   # 2‑3 sentence overall risk assessment

class ActionOutput(BaseModel):
    output_type: str  # "legal_notice", "complaint_letter", "action_plan"
    content: str       # the full generated text