# ============================================
# src/chain.py — RAG chain assembly
# Single responsibility: connect retrieval to Claude
# ============================================

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.prompts import get_prompt
from config import LLM_MODEL, LLM_TEMPERATURE, ANTHROPIC_API_KEY

def format_docs(docs: list) -> str:
    """
    Format retrieved chunks with page citations.
    This is what goes into {context} in the prompt.

    Args:
        docs: list of Document objects from retriever

    Returns:
        formatted string with page citations
    """
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Page {int(page)+1}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def create_rag_chain(retriever, language: str = "English"):
    """
    Build the complete RAG chain.

    Chain flow:
    question → {retriever fetches context, question passes through}
             → prompt template fills {context} and {question}
             → Claude generates answer
             → StrOutputParser extracts text

    Args:
        retriever: ChromaDB retriever object
        language:  response language for prompt selection

    Returns:
        Runnable chain that accepts a question string
    """
    llm = ChatAnthropic(
        model       = LLM_MODEL,
        temperature = LLM_TEMPERATURE,
        anthropic_api_key = ANTHROPIC_API_KEY,
    )

    prompt = get_prompt(language)

    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain