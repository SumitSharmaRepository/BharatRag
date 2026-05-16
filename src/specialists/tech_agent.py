# ============================================
# src/specialists/tech_agent.py
# ============================================
# Technical documentation specialist.
# Handles: Python, Streamlit, AI tools,
#          SmartDocs, APIs, deployment.
# ============================================

from langchain_core.messages import HumanMessage
from src.agents.state import MultiAgentState
from src.specialists.base import (
    llm, get_retriever,
    format_docs_with_citations,
    LANG_INSTRUCTIONS
)


def tech_agent_node(state: MultiAgentState) -> dict:
    question   = state["question"]
    user_facts = state.get("user_facts", "")
    language   = state.get("language", "English")

    print(f"  [TechAgent] Handling: '{question}'")

    docs = get_retriever(
        doc_name_filter="SmartDocs_Complete_Learning_Guide.pdf"
    ).invoke(question)

    if not docs:
        return {
            "answer":     "No technical documentation found.",
            "agent_used": "TechAgent",
            "documents":  [],
        }

    context, doc_texts = format_docs_with_citations(docs)
    lang_instruction   = LANG_INSTRUCTIONS.get(
        language, "Answer in clear English."
    )
    memory_section = f"\n{user_facts}\n" if user_facts else ""

    prompt = f"""You are a technical documentation assistant \
specialising in Python, Streamlit, and AI development.
{memory_section}
{lang_instruction}

Answer using ONLY the provided context.
Be precise. Include code references when relevant.
Cite page numbers. If not found say:
"This information is not in the technical documentation."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"  [TechAgent] Answer ready")

    return {
        "answer":     response.content,
        "agent_used": "TechAgent",
        "documents":  doc_texts,
    }