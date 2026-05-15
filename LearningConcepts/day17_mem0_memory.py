# ============================================
# DAY 17: Mem0 Persistent Memory
# ============================================
# DAY 9 RECAP (in-session memory):
# chat_history in state
# → lost when script ends
# → Type 2 memory
#
# DAY 17 (cross-session memory):
# Mem0 extracts facts from conversation
# → stored in vector database permanently
# → survives across days and sessions
# → Type 3 memory
#
# Together: complete memory system
# Type 2 = what we talked about today
# Type 3 = what I know about you always
# ============================================
import warnings
warnings.filterwarnings("ignore")
import os
import atexit
import qdrant_client
from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# ── Setup ─────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
    model             = "claude-sonnet-4-5",
    temperature       = 0,
    anthropic_api_key = ANTHROPIC_KEY,
)

# ============================================
# MEM0 CONFIGURATION
# ============================================
# Mem0 needs an LLM to extract facts
# and an embedding model to store them.
#
# Config tells Mem0:
# → Which LLM to use for fact extraction
# → Which embedding model for storage
# → Where to store facts (local by default)
# ============================================

config = {
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
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dims": 384,  # ← ADD THIS
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "bharatrag_memory",
            "embedding_model_dims": 384,  # ← ADD THIS
            "on_disk": True,
            "path": "/home/sumit/bharatrag/mem0_db",
        }
    }
}

print("Initialising Mem0...")
memory = Memory.from_config(config)
print("Mem0 ready!")

# ============================================
# CORE MEMORY OPERATIONS
# ============================================

def save_to_memory(user_id: str,
                   messages: list[dict]) -> list:
    """
    Extract and save facts from conversation.

    Mem0 reads the messages and automatically
    identifies important facts to remember.

    Args:
        user_id:  unique user identifier
                  "sumit_sharma" or "user_123"
                  Facts are stored per user
        messages: list of {role, content} dicts

    Returns:
        list of extracted memories
    """
    result = memory.add(
        messages = messages,
        user_id  = user_id,
    )
    return result.get("results", [])


def get_relevant_memories(user_id: str,
                          query: str) -> list:
    """
    Search stored memories relevant to current query.

    Mem0 embeds the query and searches stored facts
    by semantic similarity.

    Args:
        user_id: whose memories to search
        query:   current question or context

    Returns:
        list of relevant memory objects
    """
    # FIX: Move user_id into the filters dictionary
    memories = memory.search(
        query   = query,
        filters = {"user_id": user_id},
        limit   = 5,
    )
    return memories.get("results", [])


def get_all_memories(user_id: str) -> list:
    """Get ALL stored memories for a user."""

    # OLD — no longer works:
    # result = memory.get_all(user_id=user_id)

    # NEW — correct API:
    result = memory.get_all(
        filters={"user_id": user_id}
    )
    return result.get("results", [])


def format_memories_for_prompt(
    memories: list
) -> str:
    """
    Format memories as context for Claude.

    Converts list of memory objects into
    readable text for the system prompt.
    """
    if not memories:
        return ""

    facts = []
    for mem in memories:
        fact = mem.get("memory", "")
        if fact:
            facts.append(f"- {fact}")

    if not facts:
        return ""

    return "What I know about this user:\n" + \
           "\n".join(facts)


# ============================================
# BHARATRAG WITH PERSISTENT MEMORY
# ============================================

def chat_with_memory(
    user_id:  str,
    question: str,
    context:  str = "",
) -> str:
    """
    Answer a question with persistent memory.

    Flow:
    1. Search relevant memories for this query
    2. Inject memories into Claude's context
    3. Generate answer aware of user history
    4. Save this exchange to memory

    Args:
        user_id:  identifies the user
        question: current question
        context:  retrieved document chunks (RAG)

    Returns:
        Claude's answer
    """
    # Step 1: Get relevant memories
    memories     = get_relevant_memories(
        user_id, question
    )
    memory_text  = format_memories_for_prompt(memories)

    if memory_text:
        print(f"  [memory] Found {len(memories)} "
              f"relevant memories")
    else:
        print(f"  [memory] No relevant memories yet")

    # Step 2: Build prompt with memory
    system_parts = []

    if memory_text:
        system_parts.append(memory_text)

    if context:
        system_parts.append(
            f"Document context:\n{context}"
        )

    system_parts.append(
        "Answer helpfully. Use memory to personalise. "
        "If user prefers Hindi/Hinglish, use that language."
    )

    full_prompt = "\n\n".join(system_parts) + \
                  f"\n\nQuestion: {question}\nAnswer:"

    # Step 3: Generate answer
    response = llm.invoke(
        [HumanMessage(content=full_prompt)]
    )
    answer   = response.content

    # Step 4: Save this exchange to memory
    messages_to_save = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    extracted = save_to_memory(user_id, messages_to_save)

    if extracted:
        print(f"  [memory] Saved {len(extracted)} "
              f"new facts")

    return answer


# ============================================
# DEMO: Show memory working across sessions
# ============================================

def demo_memory():
    """
    Demonstrates persistent memory across
    simulated sessions.

    Session 1: User reveals preferences
    Session 2: Agent remembers without being told
    """
    USER_ID = "sumit_sharma_demo"

    # Clear old memories for clean demo
    try:
        memory.delete_all(user_id=USER_ID)
        print("Cleared old memories for clean demo")
    except Exception:
        pass

    print("\n" + "=" * 55)
    print("SESSION 1 — User reveals preferences")
    print("=" * 55)

    session1_exchanges = [
        "I am a CA firm owner in Lucknow",
        "I prefer answers in Hinglish",
        "I mostly ask about GST and income tax",
        "What is the GST filing deadline?",
    ]

    for message in session1_exchanges:
        print(f"\nUser: {message}")
        answer = chat_with_memory(
            user_id  = USER_ID,
            question = message,
        )
        print(f"Agent: {answer[:150]}...")

    # Show what was remembered
    print("\n" + "=" * 55)
    print("MEMORY STORED AFTER SESSION 1:")
    print("=" * 55)
    all_mems = get_all_memories(USER_ID)
    for i, mem in enumerate(all_mems):
        print(f"  {i+1}. {mem.get('memory', '')}")

    print("\n" + "=" * 55)
    print("SESSION 2 — New session, agent remembers")
    print("(Simulating new conversation — no history)")
    print("=" * 55)

    session2_questions = [
        # No mention of language preference
        # Agent should still use Hinglish
        "What is Section 80C?",

        # No mention of being CA
        # Agent should know context
        "What documents do my clients need?",
    ]

    for question in session2_questions:
        print(f"\nUser: {question}")
        answer = chat_with_memory(
            user_id  = USER_ID,
            question = question,
        )
        print(f"Agent: {answer[:200]}...")
        print()


# ============================================
# INTERACTIVE MODE
# ============================================

def interactive_with_memory(user_id: str = None):
    """
    Interactive chat with persistent memory.
    Memory builds up as you chat.
    """
    if not user_id:
        user_id = input(
            "Enter your user ID "
            "(or press Enter for 'default_user'): "
        ).strip() or "default_user"

    print(f"\nUser ID: {user_id}")

    # Show existing memories
    existing = get_all_memories(user_id)
    if existing:
        print(f"\nI remember {len(existing)} "
              f"things about you:")
        for mem in existing[:5]:
            print(f"  → {mem.get('memory', '')}")
    else:
        print("\nNo memories yet — "
              "I'll learn as we talk")

    print("\n" + "=" * 55)
    print("Chat with persistent memory")
    print("Type 'memories' to see what I remember")
    print("Type 'clear' to reset memories")
    print("Type 'exit' to quit")
    print("=" * 55)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        if question.lower() == "memories":
            all_mems = get_all_memories(user_id)
            print(f"\nI know {len(all_mems)} "
                  f"things about you:")
            for mem in all_mems:
                print(f"  → {mem.get('memory', '')}")
            continue

        if question.lower() == "clear":
            memory.delete_all(user_id=user_id)
            print("Memories cleared!")
            continue

        answer = chat_with_memory(
            user_id  = user_id,
            question = question,
        )
        print(f"\nAgent: {answer}")

# Suppress Qdrant shutdown error
original_del = qdrant_client.QdrantClient.__del__
def safe_del(self):
    try:
        original_del(self)
    except Exception:
        pass
qdrant_client.QdrantClient.__del__ = safe_del


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 55)
    print("Day 17: Mem0 Persistent Memory")
    print("=" * 55)

    print("\nChoose mode:")
    print("1. Demo (shows memory across sessions)")
    print("2. Interactive (chat with memory)")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        demo_memory()
    else:
        interactive_with_memory()