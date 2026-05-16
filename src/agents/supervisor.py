# ============================================
# src/agents/supervisor.py
# ============================================
# Supervisor classifies question domain
# and routes to correct specialist.
#
# Day 18 additions:
# - Reads user_facts before classifying
#   (memory helps classify better)
# - Routes to LogisticsAgent (new)
# - Arabic language support
# ============================================

from langchain_core.messages import HumanMessage
from src.agents.state import MultiAgentState
from src.specialists.base import llm
from src.memory.persistent import fetch_memories


def memory_retrieve_node(
    state: MultiAgentState
) -> dict:
    """
    Node: Fetch Mem0 memories before supervisor.

    This runs BEFORE classification so supervisor
    can use user context to classify better.

    Example:
    user_facts: "User is a logistics company owner"
    question:   "What are the charges?"
    → supervisor knows → route to LogisticsAgent
    → without facts → would go to GeneralAgent
    """
    user_id    = state.get("user_id", "default_user")
    question   = state["question"]

    print(f"  [memory] Fetching facts for: {user_id}")
    user_facts = fetch_memories(user_id, question)

    if user_facts:
        print(f"  [memory] Found relevant facts")
    else:
        print(f"  [memory] No facts yet")

    return {"user_facts": user_facts}


def supervisor_node(state: MultiAgentState) -> dict:
    """
    Supervisor: classify question into domain.

    Uses user_facts from Mem0 for better classification.
    Returns domain string used by router.
    """
    question   = state["question"]
    user_facts = state.get("user_facts", "")

    print(f"\n[Supervisor] Classifying: '{question}'")

    # Include user context in classification
    context_hint = ""
    if user_facts:
        context_hint = f"\nUser context: {user_facts}\n"

    classify_prompt = f"""Classify this question into one category.
{context_hint}
Question: "{question}"

Categories:
- technical:  Python, Streamlit, AI tools, SmartDocs,
              code, APIs, V1/V2/V3 versions, deployment
- research:   CRAG, RAG systems, research papers,
              benchmarks, academic concepts, Self-RAG
- logistics:  invoices, delivery challans, purchase orders,
              e-way bills, freight, vendors, amounts,
              GST on invoices, shipping, warehouse
- general:    anything else or unclear

Reply ONLY with one word: technical, research, \
logistics, or general"""

    response = llm.invoke(
        [HumanMessage(content=classify_prompt)]
    )
    domain   = response.content.strip().lower()

    if "technical"  in domain: domain = "technical"
    elif "research" in domain: domain = "research"
    elif "logistics"in domain: domain = "logistics"
    else:                      domain = "general"

    print(f"[Supervisor] Domain: {domain}")
    return {"agent_used": domain}


def supervisor_router(state: MultiAgentState) -> str:
    """
    Router: maps domain to specialist node name.
    """
    domain = state.get("agent_used", "general")
    routing = {
        "technical":  "tech_agent",
        "research":   "research_agent",
        "logistics":  "logistics_agent",
        "general":    "general_agent",
    }
    next_node = routing.get(domain, "general_agent")
    print(f"[Supervisor] Routing to: {next_node}")
    return next_node