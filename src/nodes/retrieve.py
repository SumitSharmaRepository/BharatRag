# ============================================
# src/nodes/retrieve.py
# ============================================
# Retrieves relevant chunks from Pinecone.
# Uses rewritten query if available (Day 7).
# Falls back to original query otherwise.
# ============================================

from src.agents.state import RAGState
from src.specialists.base import get_retriever


def retrieve_node(state: RAGState) -> dict:
    """
    Retrieve relevant chunks from Pinecone.

    Uses rewritten_question if attempts > 0
    and rewritten_question exists.
    Otherwise uses original question.

    Returns updated documents and increments attempts.
    """
    attempts  = state.get("attempts", 0)
    rewritten = state.get("rewritten_question", "")

    if attempts > 0 and rewritten:
        query = rewritten
        print(f"  [retrieve] Rewritten: '{query}'")
    else:
        query = state["question"]
        print(f"  [retrieve] Original: '{query}'")

    retriever = get_retriever(k=3)
    docs      = retriever.invoke(query)
    doc_texts = []

    for i, doc in enumerate(docs):
        page     = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get(
            "doc_name", "unknown.pdf"
        )
        print(
            f"    Chunk {i+1} "
            f"[{doc_name}, p.{int(page)+1}]: "
            f"{doc.page_content[:50]}..."
        )
        doc_texts.append(
            f"[{doc_name}, Page {int(page)+1}]\n"
            f"{doc.page_content}"
        )

    return {
        "documents": doc_texts,
        "attempts":  attempts + 1,
    }