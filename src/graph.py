import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import AgentState

from nodes.input_processor import input_processor
from nodes.jurisdiction_detector import jurisdiction_detector
from nodes.category_classifier import category_classifier
from nodes.law_retriever import law_retriever
from nodes.document_analyser import document_analyser
from nodes.legal_reasoner import legal_reasoner
# from nodes.confidence_evaluator import confidence_evaluator
from nodes.user_choice_router import user_choice_router
from nodes.action_generator import action_generator as ag_node
from nodes.escalation_handler import escalation_handler as eh_node

def has_document(state: AgentState):
    print("DEBUG has_document - document_text:", repr(state.document_text)[:100])
    if state.document_text:
        return "analyse_doc"
    return "reason"

def after_category(state: AgentState):
    if state.category and "other" in state.category:
        return "escalate"
    return "continue"

def after_user_choice(state: AgentState):
    if state.user_choice == "generate_document":
        return "action"
    return "escalate"

def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("input_processor", input_processor)
    graph.add_node("jurisdiction_detector", jurisdiction_detector)
    graph.add_node("category_classifier", category_classifier)
    graph.add_node("law_retriever", law_retriever)
    graph.add_node("document_analyser", document_analyser)
    graph.add_node("legal_reasoner", legal_reasoner)
    # graph.add_node("confidence_evaluator", confidence_evaluator)
    graph.add_node("user_choice_router", user_choice_router)
    graph.add_node("action_generator", ag_node)
    graph.add_node("escalation_handler", eh_node)

    # Edges
    graph.set_entry_point("input_processor")
    graph.add_edge("input_processor", "jurisdiction_detector")
    graph.add_edge("jurisdiction_detector", "category_classifier")

    graph.add_conditional_edges(
        "category_classifier",
        after_category,
        {
            "escalate": "escalation_handler",
            "continue": "law_retriever"
        }
    )

    graph.add_conditional_edges(
        "law_retriever",
        has_document,
        {"analyse_doc": "document_analyser", "reason": "legal_reasoner"}
    )
    graph.add_edge("document_analyser", "legal_reasoner")
    graph.add_edge("legal_reasoner", "user_choice_router")


    graph.add_conditional_edges(
        "user_choice_router",
        after_user_choice,
        {
            "action": "action_generator",
            "escalate": "escalation_handler"
        }
    )

    graph.add_edge("action_generator", END)
    graph.add_edge("escalation_handler", END)

    return graph

db_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(db_conn)
app = build_graph().compile(checkpointer=checkpointer)