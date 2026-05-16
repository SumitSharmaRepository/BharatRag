# ============================================
# src/specialists/general_agent.py
# ============================================
# Fallback specialist.
# Searches ALL documents without filter.
# Used when supervisor cannot classify.
# ============================================

from langchain_core.messages import HumanMessage
from src.agents.state import MultiAgentState
from src.specialists.base import (
    llm, get_retriever,
    format_docs_with_citations,
    LANG_INSTRUCTIONS
)


def general_agent_node(state: MultiAgentState) -> dict:
    question   = state["question"]
    user_facts = state.get("user_facts", "")
    language   = state.get("language", "English")

    print(f"  [GeneralAgent] Handling: '{question}'")

    # No filter — searches everything
    docs = get_retriever(k=4).invoke(question)

    if not docs:
        return {
            "answer":     "I could not find relevant "
                         "information in any document.",
            "agent_used": "GeneralAgent",
            "documents":  [],
        }

    context, doc_texts = format_docs_with_citations(docs)
    lang_instruction   = LANG_INSTRUCTIONS.get(
        language, "Answer in clear English."
    )
    memory_section = f"\n{user_facts}\n" if user_facts else ""

    prompt = f"""You are a helpful document assistant.
{memory_section}
{lang_instruction}

Answer using ONLY the provided context.
Cite which document your answer comes from.
If not found say:
"I could not find this information in the documents."

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