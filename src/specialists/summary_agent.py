# ============================================
# src/specialists/summary_agent.py
# ============================================
# Handles summarisation requests.
#
# What makes it different:
# → Uses k=8 (wider coverage of document)
# → Condensation-focused prompt
# → Returns structured key points
# → Language-aware summary style
#
# Examples:
# "Summarise the CRAG paper"
# "Give me key points from the SmartDocs guide"
# "What are the main topics in this document?"
# ============================================

from langchain_core.messages import HumanMessage
from src.agents.state import MultiAgentState
from src.specialists.base import (
    llm, get_retriever,
    format_docs_with_citations,
    LANG_INSTRUCTIONS
)


def summary_agent_node(
    state: MultiAgentState
) -> dict:
    """
    Specialist: Document summarisation agent.

    Key difference from other agents:
    → k=8 retrieves wider document coverage
    → Prompt focused on condensation not lookup
    → Returns executive summary + key points
    → Language-aware formatting
    """
    question   = state["question"]
    user_facts = state.get("user_facts", "")
    language   = state.get("language", "English")

    print(f"  [SummaryAgent] Handling: '{question}'")

    # Higher k = more document coverage
    # Regular agents use k=3
    # Summary needs broader context
    retriever = get_retriever(k=8)
    docs      = retriever.invoke(question)

    if not docs:
        return {
            "answer":     "I could not find enough "
                         "content to summarise.",
            "agent_used": "SummaryAgent",
            "documents":  [],
        }

    context, doc_texts = format_docs_with_citations(docs)
    lang_instruction   = LANG_INSTRUCTIONS.get(
        language, "Answer in clear English."
    )
    memory_section = f"\n{user_facts}\n" if user_facts else ""

    # Detect which document to summarise
    doc_names = list(set(
        doc.split(",")[0].strip("[").strip()
        for doc in doc_texts
        if doc.startswith("[")
    ))
    doc_hint = f"from {', '.join(doc_names)}" \
               if doc_names else ""

    prompt = f"""You are a document summarisation specialist.
{memory_section}
{lang_instruction}

Create a clear, concise summary {doc_hint}.

Structure your summary as:

**Executive Summary** (2-3 sentences)
[Brief overview of the entire document]

**Key Points**
1. [Most important point]
2. [Second important point]
3. [Third important point]
4. [Fourth important point — if relevant]
5. [Fifth important point — if relevant]

**Main Conclusion**
[What the document ultimately says or recommends]

Use ONLY the provided context.
Be concise — every sentence must add value.
Cite page numbers where relevant.

Context:
{context}

Request: {question}

Summary:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"  [SummaryAgent] Answer ready")

    return {
        "answer":     response.content,
        "agent_used": "SummaryAgent",
        "documents":  doc_texts,
    }