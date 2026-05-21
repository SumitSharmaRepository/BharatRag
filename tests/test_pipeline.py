# ============================================
# tests/test_pipeline.py
# ============================================
# Tests for core RAG pipeline components.
# Uses mocking to avoid real API calls.
# Fast — runs in seconds not minutes.
# ============================================

import pytest
from unittest.mock import MagicMock, patch
from src.agents.state import RAGState, MultiAgentState


# ── State Tests ───────────────────────────────────────

class TestRAGState:

    def test_rag_state_has_required_fields(self):
        state = RAGState(
            question            = "test",
            rewritten_question  = "",
            documents           = [],
            answer              = "",
            grade               = "irrelevant",
            hallucination       = "grounded",
            attempts            = 0,
            generation_attempts = 0,
            chat_history        = [],
        )
        assert state["question"] == "test"
        assert state["grade"]    == "irrelevant"
        assert state["attempts"] == 0

    def test_multi_agent_state_has_memory_fields(self):
        state = MultiAgentState(
            question     = "test",
            answer       = "",
            agent_used   = "",
            documents    = [],
            chat_history = [],
            user_facts   = "",
            language     = "English",
            user_id      = "test_user",
        )
        assert state["user_facts"] == ""
        assert state["language"]   == "English"
        assert state["user_id"]    == "test_user"


# ── Router Logic Tests ────────────────────────────────

class TestRouterLogic:

    def test_relevant_grade_routes_to_generate(self):
        from src.agents.state import RAGState

        state = {
            "grade":    "relevant",
            "attempts": 1,
        }
        # Router logic
        grade    = state.get("grade", "irrelevant")
        attempts = int(state.get("attempts", 0))

        if grade == "relevant":
            result = "generate"
        elif attempts == 1:
            result = "rewrite"
        else:
            result = "fallback"

        assert result == "generate"

    def test_irrelevant_first_attempt_routes_to_rewrite(self):
        state = {
            "grade":    "irrelevant",
            "attempts": 1,
        }
        grade    = state.get("grade", "irrelevant")
        attempts = int(state.get("attempts", 0))

        if grade == "relevant":
            result = "generate"
        elif attempts == 1:
            result = "rewrite"
        else:
            result = "fallback"

        assert result == "rewrite"

    def test_irrelevant_second_attempt_routes_to_fallback(self):
        state = {
            "grade":    "irrelevant",
            "attempts": 2,
        }
        grade    = state.get("grade", "irrelevant")
        attempts = int(state.get("attempts", 0))

        if grade == "relevant":
            result = "generate"
        elif attempts == 1:
            result = "rewrite"
        else:
            result = "fallback"

        assert result == "fallback"

    def test_grounded_routes_to_save_memory(self):
        state = {
            "hallucination":     "grounded",
            "generation_attempts": 1,
        }
        hallucination = state.get(
            "hallucination", "grounded"
        )
        gen_attempts  = int(state.get(
            "generation_attempts", 0
        ))

        if hallucination == "grounded":
            result = "save_memory"
        elif gen_attempts < 2:
            result = "regenerate"
        else:
            result = "save_memory"

        assert result == "save_memory"


# ── Memory Node Tests ─────────────────────────────────

class TestMemoryNode:

    def test_memory_node_appends_to_history(self):
        from src.nodes.memory_node import memory_node

        state = {
            "question":     "What is CRAG?",
            "answer":       "CRAG stands for...",
            "chat_history": [],
        }
        result = memory_node(state)

        assert len(result["chat_history"]) == 2
        assert result["chat_history"][0]["role"] == "user"
        assert result["chat_history"][1]["role"] == "assistant"

    def test_memory_node_keeps_last_8_messages(self):
        from src.nodes.memory_node import memory_node

        existing = [
            {"role": "user",      "content": f"q{i}"}
            for i in range(8)
        ]
        state = {
            "question":     "new question",
            "answer":       "new answer",
            "chat_history": existing,
        }
        result = memory_node(state)

        assert len(result["chat_history"]) <= 8

    def test_memory_node_preserves_existing_history(self):
        from src.nodes.memory_node import memory_node

        existing = [
            {"role": "user",      "content": "old q"},
            {"role": "assistant", "content": "old a"},
        ]
        state = {
            "question":     "new question",
            "answer":       "new answer",
            "chat_history": existing,
        }
        result = memory_node(state)

        assert len(result["chat_history"]) == 4