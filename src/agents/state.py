# ============================================
# src/agents/state.py
# ============================================
# Central state definition for BharatRAG agents.
# Every node reads and writes to this state.
#
# Evolution:
# Day 6:  question, documents, answer, grade, attempts
# Day 7:  + rewritten_question
# Day 8:  + hallucination, generation_attempts
# Day 9:  + chat_history
# Day 10: + agent_used
# Day 18: + user_facts, language, user_id (NEW)
# ============================================

from typing import TypedDict, List, Optional


class RAGState(TypedDict):
    """
    Single-agent RAG state.
    Used by Days 6-9 pipeline.
    """
    question:            str
    rewritten_question:  str
    documents:           List[str]
    answer:              str
    grade:               str        # "relevant" or "irrelevant"
    hallucination:       str        # "grounded" or "hallucinated"
    attempts:            int        # retrieval retry counter
    generation_attempts: int        # generation retry counter
    chat_history:        List[dict] # in-session memory


class MultiAgentState(TypedDict):
    """
    Multi-agent state with persistent memory.
    Used by Day 10+ supervisor + specialists.

    New in Day 18:
    user_facts: Mem0 memories injected before supervisor
    language:   explicit language selection
    user_id:    identifies user for Mem0 storage
    """
    question:     str
    answer:       str
    agent_used:   str        # which specialist handled it
    documents:    List[str]  # retrieved chunks
    chat_history: List[dict] # in-session memory (Day 9)
    user_facts:   str        # Mem0 persistent facts (Day 17)
    language:     str        # English/Hindi/Hinglish/Arabic
    user_id:      str        # unique user identifier


# ── Language instructions ──────────────────────────────
# Used by all agents and generate nodes.
# Add new languages here only — nothing else changes.
LANGUAGE_INSTRUCTIONS = {
    "English":       "Answer in clear English.",
    "Hindi / हिंदी": "हमेशा हिंदी में जवाब दें।",
    "Hinglish":      "Answer in Hinglish — natural mix "
                    "of Hindi and English.",
    "Arabic / عربي": "أجب باللغة العربية بوضوح.",
}


def get_language_instruction(language: str) -> str:
    """Return language instruction for prompt."""
    return LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS["English"]
    )


def initial_multi_agent_state(
    question: str,
    user_id:  str  = "default_user",
    language: str  = "English",
    chat_history: List[dict] = None,
) -> MultiAgentState:
    """
    Create clean initial state for one agent run.
    Convenience function used by api.py and main.py.
    """
    return {
        "question":     question,
        "answer":       "",
        "agent_used":   "",
        "documents":    [],
        "chat_history": chat_history or [],
        "user_facts":   "",
        "language":     language,
        "user_id":      user_id,
    }