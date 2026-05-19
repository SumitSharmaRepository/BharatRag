from langchain_core.messages import HumanMessage
from src.agents.state import RAGState
from src.specialists.base import llm_fast  # ← Haiku

def hallucination_check_node(state: RAGState) -> dict:
    """
    Hallucination check — uses Haiku (cheap).
    Compares answer to context — simple task.
    """
    answer    = state["answer"]
    documents = state["documents"]
    question  = state["question"]

    print(f"  [hallucination] Verifying (Haiku)...")

    prompt = f"""Is this answer supported by the context?

Context:
{chr(10).join(documents[:2])}

Answer: {answer[:500]}

Reply ONLY: grounded or hallucinated"""

    response      = llm_fast.invoke(  # ← Haiku
        [HumanMessage(content=prompt)]
    )
    result        = response.content.strip().lower()
    hallucination = "hallucinated" \
                    if "hallucinated" in result \
                    else "grounded"

    print(f"  [hallucination] Result: {hallucination}")
    return {"hallucination": hallucination}