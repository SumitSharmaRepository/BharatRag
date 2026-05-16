# ============================================
# src/agents/graph.py
# ============================================
# Builds the complete multi-agent graph.
# Connects all nodes and specialists.
#
# Graph flow:
# START
#   → memory_retrieve (fetch Mem0 facts)
#   → supervisor (classify domain)
#   → [router] → specialist
#   → memory_save (persist new facts)
#   → END
# ============================================

import os
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from src.agents.state import MultiAgentState
from src.agents.supervisor import (
    memory_retrieve_node,
    supervisor_node,
    supervisor_router,
)
from src.specialists.tech_agent      import tech_agent_node
from src.specialists.research_agent  import research_agent_node
from src.specialists.logistics_agent import logistics_agent_node
from src.specialists.general_agent   import general_agent_node
from src.memory.persistent           import save_memories


def memory_save_node(state: MultiAgentState) -> dict:
    """
    Node: Save Q&A to Mem0 after answer.
    Runs after specialist generates answer.
    Extracts facts for future sessions.
    """
    user_id  = state.get("user_id", "default_user")
    question = state["question"]
    answer   = state["answer"]

    save_memories(user_id, question, answer)
    return {}


def build_multi_agent(with_memory: bool = True):
    """
    Build and compile the multi-agent graph.

    Args:
        with_memory: include Mem0 nodes
                     set False for testing without Mem0

    Graph structure:
    memory_retrieve → supervisor → [router]
                                 → tech_agent      → memory_save → END
                                 → research_agent  → memory_save → END
                                 → logistics_agent → memory_save → END
                                 → general_agent   → memory_save → END
    """
    workflow = StateGraph(MultiAgentState)

    # ── Register nodes ────────────────────────────────
    if with_memory:
        workflow.add_node("memory_retrieve",  memory_retrieve_node)
    workflow.add_node("supervisor",       supervisor_node)
    workflow.add_node("tech_agent",       tech_agent_node)
    workflow.add_node("research_agent",   research_agent_node)
    workflow.add_node("logistics_agent",  logistics_agent_node)
    workflow.add_node("general_agent",    general_agent_node)
    if with_memory:
        workflow.add_node("memory_save",  memory_save_node)

    # ── Fixed edges ───────────────────────────────────
    if with_memory:
        workflow.set_entry_point("memory_retrieve")
        workflow.add_edge("memory_retrieve", "supervisor")
    else:
        workflow.set_entry_point("supervisor")

    # ── Conditional routing from supervisor ───────────
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "tech_agent":      "tech_agent",
            "research_agent":  "research_agent",
            "logistics_agent": "logistics_agent",
            "general_agent":   "general_agent",
        }
    )

    # ── All specialists → memory_save → END ──────────
    if with_memory:
        for specialist in [
            "tech_agent", "research_agent",
            "logistics_agent", "general_agent"
        ]:
            workflow.add_edge(specialist, "memory_save")
        workflow.add_edge("memory_save", END)
    else:
        for specialist in [
            "tech_agent", "research_agent",
            "logistics_agent", "general_agent"
        ]:
            workflow.add_edge(specialist, END)

    return workflow.compile()