# ============================================
# src/memory/persistent.py
# ============================================
# Mem0 cross-session memory.
# Extracts and stores user facts permanently.
#
# Type 3 memory — survives across sessions.
# Contrast with chat_history (Type 2) which
# resets when the script ends.
# ============================================

import os
from mem0 import Memory

ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MEM0_DB_PATH    = "/home/sumit/bharatrag/mem0_db"

# ── Mem0 config ───────────────────────────────────────
# Uses Haiku for fact extraction (cheap)
# Uses HuggingFace for embeddings (free)
# dims=384 MUST match embedding model
_config = {
    "llm": {
        "provider": "anthropic",
        "config": {
            "model":   "claude-haiku-4-5-20251001",
            "api_key": ANTHROPIC_KEY,
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model":          EMBEDDING_MODEL,
            "embedding_dims": 384,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name":      "bharatrag_memory",
            "embedding_model_dims": 384,
            "on_disk":              True,
            "path":                 MEM0_DB_PATH,
        }
    }
}

# Singleton — initialised once at import
_memory = None

def get_memory() -> Memory:
    """Get or create Mem0 instance."""
    global _memory
    if _memory is None:
        _memory = Memory.from_config(_config)
    return _memory


def fetch_memories(user_id: str,
                   query: str) -> str:
    """
    Search relevant memories for this query.
    Returns formatted string for prompt injection.

    Args:
        user_id: identifies the user
        query:   current question (used for search)

    Returns:
        formatted memory string or empty string
    """
    try:
        mem    = get_memory()
        result = mem.search(
            query   = query,
            filters={"user_id": user_id},
            limit   = 5,
        )
        facts = [
            r.get("memory", "")
            for r in result.get("results", [])
            if r.get("memory")
        ]
        if not facts:
            return ""
        return "User context:\n" + \
               "\n".join(f"- {f}" for f in facts)
    except Exception as e:
        print(f"  [memory] fetch failed: {e}")
        return ""


def save_memories(user_id: str,
                  question: str,
                  answer:   str) -> int:
    """
    Extract and save facts from Q&A exchange.

    Args:
        user_id:  identifies the user
        question: what user asked
        answer:   what agent answered

    Returns:
        number of new facts saved
    """
    try:
        mem    = get_memory()
        result = mem.add(
            messages = [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
            user_id = user_id,
        )
        saved = len(result.get("results", []))
        if saved:
            print(f"  [memory] Saved {saved} new facts")
        return saved
    except Exception as e:
        print(f"  [memory] save failed: {e}")
        return 0


def clear_memories(user_id: str) -> None:
    """Clear all memories for a user."""
    try:
        get_memory().delete_all(
            filters={"user_id": user_id}
        )
        print(f"  [memory] Cleared memories for {user_id}")
    except Exception as e:
        print(f"  [memory] clear failed: {e}")