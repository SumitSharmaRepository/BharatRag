from langchain_core.messages import HumanMessage
from src.agents.state import RAGState
from src.specialists.base import llm_fast  # ← Haiku

def grade_node(state: RAGState) -> dict:
    """
    Grade relevance — uses Haiku (cheap).
    Simple yes/no decision does not need Sonnet.
    """
    question  = state["question"]
    documents = state["documents"]

    print(f"  [grade] Checking relevance (Haiku)...")

    prompt = f"""Are these documents relevant to: "{question}"?

Documents:
{chr(10).join(documents[:2])}

Reply ONLY: relevant or irrelevant"""

    response = llm_fast.invoke(  # ← Haiku
        [HumanMessage(content=prompt)]
    )
    grade  = response.content.strip().lower()
    result = "irrelevant" if "irrelevant" in grade \
             else "relevant"

    print(f"  [grade] Result: {result}")
    return {"grade": result}