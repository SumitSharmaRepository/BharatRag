# ============================================
# src/specialists/comparison_agent.py
# ============================================
# Handles cross-document comparison questions.
#
# What makes it different from other agents:
# → Retrieves from MULTIPLE documents separately
# → Structures a side-by-side answer
# → Prompt forces comparison format
#
# Examples:
# "Compare V1 and V2 of SmartDocs"
# "How does CRAG differ from standard RAG?"
# "Compare invoice INV-001 and INV-002"
# ============================================

from langchain_core.messages import HumanMessage
from src.agents.state import MultiAgentState
from src.specialists.base import (
    llm, get_retriever,
    format_docs_with_citations,
    LANG_INSTRUCTIONS
)


def _detect_entities(question: str) -> list[str]:
    """
    Extract things being compared from the question.
    Simple heuristic — looks for 'and', 'vs', 'versus'.

    Examples:
    "Compare V1 and V2" → ["V1", "V2"]
    "CRAG vs RAG"       → ["CRAG", "RAG"]
    """
    q = question.lower()

    # Split on comparison keywords
    for sep in [" vs ", " versus ", " and ", " or "]:
        if sep in q:
            parts = question.split(sep, 1)
            if len(parts) == 2:
                # Clean up common prefixes
                left  = parts[0].replace(
                    "compare ", ""
                ).replace("Compare ", "").strip()
                right = parts[1].strip().rstrip("?")
                return [left, right]

    return []


def comparison_agent_node(
    state: MultiAgentState
) -> dict:
    """
    Specialist: Cross-document comparison agent.

    Strategy:
    1. Detect what is being compared
    2. Run separate retrieval for each entity
    3. Combine context with clear labels
    4. Force structured side-by-side answer

    Falls back to single retrieval if entities
    cannot be detected.
    """
    question   = state["question"]
    user_facts = state.get("user_facts", "")
    language   = state.get("language", "English")

    print(f"  [ComparisonAgent] Handling: '{question}'")

    entities = _detect_entities(question)

    if entities and len(entities) == 2:
        # Retrieve separately for each entity
        print(f"  [ComparisonAgent] Comparing: "
              f"{entities[0]} vs {entities[1]}")

        retriever = get_retriever(k=4)

        docs_a = retriever.invoke(
            f"{entities[0]} {question}"
        )
        docs_b = retriever.invoke(
            f"{entities[1]} {question}"
        )

        _, texts_a = format_docs_with_citations(docs_a)
        _, texts_b = format_docs_with_citations(docs_b)

        context = (
            f"=== {entities[0].upper()} ===\n"
            + "\n".join(texts_a)
            + f"\n\n=== {entities[1].upper()} ===\n"
            + "\n".join(texts_b)
        )
        all_docs = texts_a + texts_b

    else:
        # Fallback: single retrieval
        print(f"  [ComparisonAgent] Single retrieval "
              f"(entities not detected)")
        retriever  = get_retriever(k=5)
        docs       = retriever.invoke(question)
        context, all_docs = format_docs_with_citations(docs)

    if not all_docs:
        return {
            "answer":     "I could not find enough "
                         "information to make a comparison.",
            "agent_used": "ComparisonAgent",
            "documents":  [],
        }

    lang_instruction = LANG_INSTRUCTIONS.get(
        language, "Answer in clear English."
    )
    memory_section = f"\n{user_facts}\n" if user_facts else ""

    prompt = f"""You are a document comparison specialist.
{memory_section}
{lang_instruction}

Your job is to compare clearly and fairly.
Structure your answer as:

**[Item A]**
- Key point 1
- Key point 2

**[Item B]**
- Key point 1
- Key point 2

**Key differences**
- Difference 1
- Difference 2

Use ONLY the provided context.
Cite document names and page numbers.
If a point cannot be compared, say so clearly.

Context:
{context}

Question: {question}

Comparison:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"  [ComparisonAgent] Answer ready")

    return {
        "answer":     response.content,
        "agent_used": "ComparisonAgent",
        "documents":  all_docs,
    }