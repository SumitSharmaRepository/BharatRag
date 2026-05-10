# ============================================
# DAY 10: Multi-Agent Supervisor Pattern
# ============================================
# DAY 9 RECAP:
# Single conversational agent with memory
# One agent handles all document types
# No specialisation
#
# DAY 10 ADDS:
# Supervisor agent routes to specialist agents
# TechAgent    → technical documents (SmartDocs)
# ResearchAgent → research papers (CRAG PDF)
# GeneralAgent  → fallback for anything else
#
# WHY SPECIALISTS?
# A CA firm's tax agent should be optimised
# for tax language, not general Q&A.
# A research agent should cite papers properly.
# Specialists give better answers in their domain.
# ============================================

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "bharatrag"
import shutil
from dotenv import load_dotenv
load_dotenv()




# LangSmith tracing — reads from .env automatically
# load_dotenv() already loaded them
# Just verify they're set
print(f"LangSmith tracing: "
      f"{os.getenv('LANGSMITH_TRACING')}")
print(f"LangSmith project: "
      f"{os.getenv('LANGSMITH_PROJECT')}")

from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# ── Setup ─────────────────────────────────────────────
CHROMA_PATH     = "/home/sumit/bharatrag/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

def get_retriever(filter_dict=None):
    """
    Get retriever with optional metadata filter.

    filter_dict lets us restrict search to
    specific documents by name.

    Example:
    filter_dict={"doc_name": "CRAG.pdf"}
    → only searches CRAG chunks
    """
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    if filter_dict:
        return vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": filter_dict
            }
        )
    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    anthropic_api_key=ANTHROPIC_KEY
)

# ============================================
# SHARED STATE
# ============================================
# All agents share the same state structure
# Supervisor fills "question"
# Specialist fills "answer" and "agent_used"
# ============================================

class MultiAgentState(TypedDict):
    question:    str
    answer:      str
    agent_used:  str        # which specialist handled it
    documents:   List[str]  # retrieved chunks
    chat_history: List[dict]

# ============================================
# SPECIALIST AGENTS
# Each is a simple retrieve + generate pattern
# No hallucination check for simplicity today
# Day 11+ adds that back
# ============================================

def tech_agent_node(state: MultiAgentState) -> dict:
    """
    Specialist: Technical documents agent.
    Optimised for SmartDocs learning guide.
    Knows about Python, Streamlit, Claude API.
    """
    question = state["question"]
    print(f"  [TechAgent] Handling: '{question}'")

    # Filter to SmartDocs document only
    retriever = get_retriever({
        "doc_name": "SmartDocs_Complete_Learning_Guide.pdf"
    })
    docs      = retriever.invoke(question)
    doc_texts = []

    for doc in docs:
        page = doc.metadata.get("page", "?")
        doc_texts.append(
            f"[SmartDocs Guide, Page {int(page)+1}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(doc_texts)

    prompt = f"""You are a technical documentation assistant
specialising in Python, Streamlit, and AI development.

Answer using ONLY the provided context.
Be precise and technical. Include code references
when relevant. Cite page numbers.

Context:
{context}

Question: {question}

Technical answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"  [TechAgent] Answer ready")
    return {
        "answer":     response.content,
        "agent_used": "TechAgent",
        "documents":  doc_texts,
    }


def research_agent_node(state: MultiAgentState) -> dict:
    """
    Specialist: Research paper agent.
    Optimised for academic content like CRAG PDF.
    Knows how to cite research papers properly.
    """
    question = state["question"]
    print(f"  [ResearchAgent] Handling: '{question}'")

    # Filter to CRAG document only
    retriever = get_retriever({
        "doc_name": "CRAG.pdf"
    })
    docs      = retriever.invoke(question)
    doc_texts = []

    for doc in docs:
        page = doc.metadata.get("page", "?")
        doc_texts.append(
            f"[CRAG Research Paper, Page {int(page)+1}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(doc_texts)

    prompt = f"""You are a research paper analysis assistant.

Answer using ONLY the provided research paper context.
Use academic language. Cite sections and page numbers.
Explain technical concepts clearly.
If methodology or results are mentioned, highlight them.

Context:
{context}

Question: {question}

Research-based answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"  [ResearchAgent] Answer ready")
    return {
        "answer":     response.content,
        "agent_used": "ResearchAgent",
        "documents":  doc_texts,
    }


def general_agent_node(state: MultiAgentState) -> dict:
    """
    Fallback: General agent.
    Searches ALL documents without filter.
    Used when supervisor cannot classify the question.
    """
    question = state["question"]
    print(f"  [GeneralAgent] Handling: '{question}'")

    retriever = get_retriever()  # no filter
    docs      = retriever.invoke(question)
    doc_texts = []

    for doc in docs:
        page     = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get(
            "doc_name", "unknown"
        )
        doc_texts.append(
            f"[{doc_name}, Page {int(page)+1}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(doc_texts)

    if not doc_texts:
        return {
            "answer":     "I could not find relevant "
                         "information in any document.",
            "agent_used": "GeneralAgent",
            "documents":  [],
        }

    prompt = f"""You are a helpful document assistant.

Answer using ONLY the provided context.
Cite which document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"  [GeneralAgent] Answer ready")
    return {
        "answer":     response.content,
        "agent_used": "GeneralAgent",
        "documents":  doc_texts,
    }


# ============================================
# SUPERVISOR
# ============================================
# Supervisor reads the question and decides
# which specialist should handle it.
#
# Uses Claude to classify the question domain.
# Returns the name of the specialist node.
# ============================================

def supervisor_node(state: MultiAgentState) -> dict:
    """
    Supervisor: classifies question and routes to specialist.

    Does NOT answer the question itself.
    Only decides WHO should answer it.

    This separation of concerns is key:
    → Supervisor = routing intelligence
    → Specialists = domain knowledge
    """
    question = state["question"]
    print(f"\n[Supervisor] Classifying: '{question}'")

    classify_prompt = f"""Classify this question:

Question: "{question}"

Categories:
- technical: Python, Streamlit, SmartDocs, Claude API,
             session state, error handling, deployment,
             code, V1, V2, V3, versions of SmartDocs
- research:  CRAG, RAG systems, benchmarks, papers,
             academic, Self-RAG, retrieval methods
- general:   anything else

Reply ONLY: technical, research, or general"""

    response   = llm.invoke(
        [HumanMessage(content=classify_prompt)]
    )
    domain     = response.content.strip().lower()

    # Clean up response
    if "technical" in domain:
        domain = "technical"
    elif "research" in domain:
        domain = "research"
    else:
        domain = "general"

    print(f"[Supervisor] Domain: {domain}")
    return {"agent_used": domain}


def supervisor_router(state: MultiAgentState) -> str:
    """
    Router: reads domain set by supervisor_node
    and returns the name of next node to execute.
    """
    domain = state.get("agent_used", "general")

    routing = {
        "technical": "tech_agent",
        "research":  "research_agent",
        "general":   "general_agent",
    }

    next_node = routing.get(domain, "general_agent")
    print(f"[Supervisor] Routing to: {next_node}")
    return next_node


# ============================================
# BUILD MULTI-AGENT GRAPH
# ============================================

def build_multi_agent():
    """
    Build supervisor + specialist graph.

    Structure:
    START → supervisor → [router]
                              ↓           ↓           ↓
                        tech_agent  research_agent  general_agent
                              ↓           ↓           ↓
                             END         END         END
    """
    workflow = StateGraph(MultiAgentState)

    # Register supervisor and all specialists
    workflow.add_node("supervisor",     supervisor_node)
    workflow.add_node("tech_agent",     tech_agent_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("general_agent",  general_agent_node)

    # Fixed edge: start at supervisor always
    workflow.set_entry_point("supervisor")

    # Conditional edge: supervisor routes to specialist
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "tech_agent":     "tech_agent",
            "research_agent": "research_agent",
            "general_agent":  "general_agent",
        }
    )

    # All specialists end the graph
    workflow.add_edge("tech_agent",     END)
    workflow.add_edge("research_agent", END)
    workflow.add_edge("general_agent",  END)

    return workflow.compile()


# ============================================
# RUN
# ============================================

def run_multi_agent(question: str, agent,
                    chat_history: list = None) -> dict:
    """Run the multi-agent system with one question."""
    print(f"\nQuestion: '{question}'")
    print("=" * 55)

    initial_state = {
        "question":     question,
        "answer":       "",
        "agent_used":   "",
        "documents":    [],
        "chat_history": chat_history or [],
    }

    final_state = agent.invoke(initial_state)

    print(f"\nAnswer [{final_state['agent_used']}]:")
    print(final_state["answer"])
    return final_state


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("Building multi-agent system...")
    agent = build_multi_agent()
    print("Multi-agent system ready!")
    print()

    # Test questions — each should route differently
    questions = [
        # Should go to TechAgent
        "What is session state in Streamlit?",

        # Should go to ResearchAgent
        "What is CRAG and how does it improve RAG?",

        # Should go to ResearchAgent
        "What benchmarks did CRAG outperform?",

        # Should go to TechAgent
        "What is the difference between V1 and V2?",

        # Should go to GeneralAgent (unclear)
        "What is the best approach for documents?",
    ]

    for q in questions:
        result = run_multi_agent(q, agent)
        print(f"\n[Routed to: {result['agent_used']}]")
        print()

    # Interactive mode
    print("=" * 55)
    print("Interactive — type 'exit' to quit")
    print("=" * 55)

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ["exit", "quit", ""]:
            break
        result = run_multi_agent(q, agent)
        print(f"\n[Routed to: {result['agent_used']}]")

"""
Building multi-agent system...
Multi-agent system ready!


Question: 'What is session state in Streamlit?'
=======================================================

[Supervisor] Classifying: 'What is session state in Streamlit?'
[Supervisor] Domain: technical
[Supervisor] Routing to: tech_agent
  [TechAgent] Handling: 'What is session state in Streamlit?'
  [TechAgent] Answer ready

Answer [TechAgent]:
# Session State in Streamlit

## Definition

`st.session_state` is a **dictionary that persists across reruns** [Page 2]. It is the mechanism by which data survives between user interactions in Streamlit applications.

## Technical Context

Due to Streamlit's rerun model, every user action (click, upload, type) causes the **entire Python file to rerun from line 1**, which means **normal variables reset every rerun** [Page 2]. Only `st.session_state` survives across these reruns.

## Conceptual Model

Think of it as: "every button click creates a brand new Python process that runs your whole file again" [Page 2]. Session state is the only data structure that persists between these process-like reruns.

## Usage Pattern

Adding a new session variable equals "adding one line to the dictionary" [Page 9]. The typical pattern is to check if a key doesn't already exist in session state before initializing it.

## Practical Application

In the SmartDocs context, `st.session_state` is specifically used to remember:
- Uploaded PDF files
- Chat history between questions

This allows the application to maintain state across user interactions [Page 2].

[Routed to: TechAgent]


Question: 'What is CRAG and how does it improve RAG?'
=======================================================

[Supervisor] Classifying: 'What is CRAG and how does it improve RAG?'
[Supervisor] Domain: research
[Supervisor] Routing to: research_agent
  [ResearchAgent] Handling: 'What is CRAG and how does it improve RAG?'
  [ResearchAgent] Answer ready

Answer [ResearchAgent]:
# CRAG: A Corrective Approach to Retrieval-Augmented Generation

## Definition and Core Concept

CRAG (Corrective Retrieval-Augmented Generation) is a plug-and-play framework that enhances RAG systems from a "corrective perspective" (Page 10). The framework is designed to be seamlessly integrated with various RAG-based approaches without requiring extensive modifications to the underlying architecture.

## Key Improvements Over Standard RAG

CRAG demonstrates significant performance improvements over both standard RAG and state-of-the-art Self-RAG across multiple benchmarks (Page 2). The framework was experimentally validated on four datasets:
- PopQA (Mallen et al., 2023)
- Biography (Min et al., 2023)
- Pub Health (Zhang et al., 2023a)
- Arc-Challenge (Bhakthavatsalam et al., 2021)

## Methodological Advantages

A critical advantage of CRAG over competing approaches like Self-RAG lies in its implementation requirements. Unlike Self-RAG, which "needs to be instruction-tuned using human or LLM annotated data to learn to output special critic tokens," CRAG "does not have any requirements for this ability" (Page 8). This makes CRAG more adaptable to future LLMs, as it can be "coupled with CRAG easily, while additional instruction tuning is still necessary for Self-RAG" (Page 8).

## Technical Implementation

The framework does require "fine-tuning an external retrieval evaluator" (Page 10), which the authors acknowledge as a limitation for future work, specifically noting the goal to "eliminate this external evaluator and equip LLMs with better retrieval evaluation capabilities" (Page 10).

[Routed to: ResearchAgent]


Question: 'What benchmarks did CRAG outperform?'
=======================================================

[Supervisor] Classifying: 'What benchmarks did CRAG outperform?'
[Supervisor] Domain: research
[Supervisor] Routing to: research_agent
  [ResearchAgent] Handling: 'What benchmarks did CRAG outperform?'
  [ResearchAgent] Answer ready

Answer [ResearchAgent]:
Based on the evaluation results presented in Table 1 (Page 7), CRAG demonstrated competitive performance across four datasets in the test sets, though the specific dataset names are partially visible in the context provided.

## Performance Comparison

According to Table 1 (Page 7), CRAG achieved the following scores across the four benchmarks:
- Dataset 1: 59.8
- Dataset 2: 74.1
- Dataset 3: 75.6
- Dataset 4: 68.6

## Outperformance of Baseline Methods

The paper explicitly states that CRAG outperformed "the standard RAG on several benchmarks" (Page 8). This superiority is attributed to a fundamental methodological advantage: CRAG does not require instruction-tuning with human or LLM-annotated data to learn special critic tokens, unlike Self-RAG, which necessitates this additional training step.

## Comparison with Self-RAG

When comparing CRAG to Self-RAG across the visible metrics in Table 1 (Page 7):
- CRAG outperformed Self-RAG on Dataset 1 (59.8 vs. 54.9)
- Self-RAG achieved higher performance on Dataset 2 (81.2 vs. 74.1)
- CRAG showed superior results on Dataset 3 (75.6 vs. 72.4)
- CRAG performed better on Dataset 4 (68.6 vs. 67.3)

The paper emphasizes CRAG's practical advantage in terms of adaptability, noting that "when more advanced LLMs are available in the future, they can be coupled with CRAG easily, while additional instruction tuning is still necessary for Self-RAG" (Page 8).

[Routed to: ResearchAgent]


Question: 'What is the difference between V1 and V2?'
=======================================================

[Supervisor] Classifying: 'What is the difference between V1 and V2?'
[Supervisor] Domain: general
[Supervisor] Routing to: general_agent
  [GeneralAgent] Handling: 'What is the difference between V1 and V2?'
  [GeneralAgent] Answer ready

Answer [GeneralAgent]:
Based on the provided documents from **SmartDocs_Complete_Learning_Guide.pdf**:

**V1 (Prototype)** is the minimum working version with approximately 50 lines of code that provides core functionality: upload a PDF, ask a question, and get an answer from Claude. However, it has **no error handling, no security, and no polish** - it's just a proof of concept that is **not for real users**. (Page 1 and Page 3)

**V2 (Production Ready)** transforms the prototype into something real users can actually use by adding three major features (Page 8):
1. **Proper error handling** - every API error shows a friendly message instead of crashing with a red screen
2. **Client API key input** - clients enter their own Anthropic key so they pay their own bills instead of you paying
3. **File validation** - bad uploads are caught gracefully
4. **UI polish**

In summary, V1 is a working prototype to prove the concept works, while V2 adds the essential production features (error handling, security, and client billing) needed for real users.

[Routed to: GeneralAgent]


Question: 'What is the best approach for documents?'
=======================================================

[Supervisor] Classifying: 'What is the best approach for documents?'
[Supervisor] Domain: general
[Supervisor] Routing to: general_agent
  [GeneralAgent] Handling: 'What is the best approach for documents?'
  [GeneralAgent] Answer ready

Answer [GeneralAgent]:
Based on the provided context from CRAG.pdf, Page 2, the best approach for documents is **not to treat complete documents equally as reference knowledge**. 

The document states that "current methods mostly treat complete documents as reference knowledge both during retrieval and utilization. But a considerable portion of the text within these retrieved documents is often non-essential for generation, which should not have been equally referred to and involved in RAG."

The paper advocates for a more selective approach that addresses scenarios where retrievers return inaccurate results, rather than indiscriminately incorporating all retrieved documents regardless of their relevance.

**Source: CRAG.pdf, Page 2**

[Routed to: GeneralAgent]

=======================================================
Interactive — type 'exit' to quit
=======================================================
"""        
