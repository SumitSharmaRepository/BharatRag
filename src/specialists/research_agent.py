# ============================================
# src/specialists/research_agent.py
# ============================================
# Research paper specialist.
# Handles: CRAG, RAG systems, benchmarks,
#          academic concepts.
# ============================================

from langchain_core.messages import HumanMessage
from src.agents.state import MultiAgentState
from src.specialists.base import (
    llm, get_retriever,
    format_docs_with_citations,
    LANG_INSTRUCTIONS
)


def research_agent_node(state: MultiAgentState) -> dict:
    question   = state["question"]
    user_facts = state.get("user_facts", "")
    language   = state.get("language", "English")

    print(f"  [ResearchAgent] Handling: '{question}'")

    docs = get_retriever(
        doc_name_filter="CRAG.pdf"
    ).invoke(question)

    if not docs:
        return {
            "answer":     "No research papers found.",
            "agent_used": "ResearchAgent",
            "documents":  [],
        }

    context, doc_texts = format_docs_with_citations(docs)
    lang_instruction   = LANG_INSTRUCTIONS.get(
        language, "Answer in clear English."
    )
    memory_section = f"\n{user_facts}\n" if user_facts else ""

    prompt = f"""You are a research paper analysis assistant.
{memory_section}
{lang_instruction}

Answer using ONLY the provided research context.
Use academic language. Cite page numbers.
Highlight methodology and results when mentioned.
If not found say:
"This information is not in the research papers."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"  [ResearchAgent] Answer ready")

    return {
        "answer":     response.content,
        "agent_used": "ResearchAgent",
        "documents":  doc_texts,
    }