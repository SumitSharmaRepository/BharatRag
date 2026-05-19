from langchain_core.messages import HumanMessage
from src.agents.state import RAGState
from src.specialists.base import llm_fast  # ← Haiku

def query_rewrite_node(state: RAGState) -> dict:
    """
    Query rewriting — uses Haiku (cheap).
    Simple rewriting task.
    """
    original  = state["question"]
    documents = state["documents"]

    print(f"  [rewrite] Rewriting (Haiku)...")

    prompt = f"""Rewrite this failed search query \
to find better results.

Original: "{original}"
Retrieved but irrelevant: {documents[0][:100] if documents else 'nothing'}

Write a better search query under 15 words.
Just the query, nothing else."""

    response  = llm_fast.invoke(  # ← Haiku
        [HumanMessage(content=prompt)]
    )
    rewritten = response.content.strip().strip('"')
    print(f"  [rewrite] New: '{rewritten}'")
    return {"rewritten_question": rewritten}