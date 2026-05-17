# ============================================
# src/specialists/base.py
# ============================================
# Shared utilities for all specialist agents.
# Avoids code duplication across specialists.
# ============================================

import os
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.retrieval.hybrid import hybrid_search, smart_alpha
from langchain_core.documents import Document

ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "bharatrag")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# ── Shared instances ──────────────────────────────────
# Created once, reused by all specialists

llm = ChatAnthropic(
    model             = "claude-sonnet-4-5",
    temperature       = 0,
    anthropic_api_key = ANTHROPIC_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ── Language instructions ─────────────────────────────
LANG_INSTRUCTIONS = {
    "English":       "Answer in clear English.",
    "Hindi / हिंदी": "हिंदी में जवाब दें।",
    "Hinglish":      "Answer in Hinglish naturally.",
    "Arabic / عربي": "أجب باللغة العربية بوضوح.",
}


def get_retriever(doc_name_filter: str = None,
                  k: int = 3):
    """
    Get Pinecone retriever with optional filter.

    Args:
        doc_name_filter: exact doc_name to filter to
                        None = search all documents
        k: number of chunks to retrieve

    Returns:
        LangChain retriever
    """
    vectorstore = PineconeVectorStore(
        index_name = PINECONE_INDEX,
        embedding  = embeddings,
    )

    if doc_name_filter:
        return vectorstore.as_retriever(
            search_kwargs={
                "k":      k,
                "filter": {"doc_name": doc_name_filter}
            }
        )
    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
def get_hybrid_retriever(
    query:       str,
    k:           int  = 3,
    filter_dict: dict = None,
) -> list[Document]:
    """
    Hybrid retriever — automatically picks alpha.
    Drop-in replacement for get_retriever().

    Usage in any specialist:
    docs = get_hybrid_retriever(question, k=3)
    instead of:
    docs = get_retriever().invoke(question)
    """
    alpha = smart_alpha(query)
    return hybrid_search(
        query       = query,
        k           = k,
        alpha       = alpha,
        filter_dict = filter_dict,
    )

def format_docs_with_citations(docs: list) -> tuple:
    """
    Format retrieved docs with source citations.

    Returns:
        tuple: (context_string, doc_texts_list)
    """
    doc_texts = []
    for doc in docs:
        page     = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get("doc_name", "unknown")
        doc_texts.append(
            f"[{doc_name}, Page {int(page)+1}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(doc_texts), doc_texts