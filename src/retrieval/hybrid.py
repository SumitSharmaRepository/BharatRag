# ============================================
# src/retrieval/hybrid.py
# ============================================
# Hybrid search = BM25 (sparse) + dense vectors
#
# Why this matters:
# Dense alone: misses exact terms
# BM25 alone:  misses semantic meaning
# Hybrid:      catches both
#
# Critical for Indian documents:
# → "Section 80C" exact match needed
# → "GST GSTIN 09AABCS1429B1ZX" exact match
# → "INV-2024-001" invoice number exact match
# → Dense search alone misses all of these
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document

from src.embeddings import get_embeddings

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "bharatrag")

# ── Shared instances ──────────────────────────────────
embeddings = get_embeddings()

pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ── BM25 Encoder ──────────────────────────────────────
# BM25 = Best Match 25
# Keyword-based relevance scoring algorithm
# Used in traditional search engines before neural search
# We use it alongside dense vectors for hybrid
_bm25 = None

def get_bm25() -> BM25Encoder:
    """
    Get or create BM25 encoder.
    Uses default parameters — works for most text.
    """
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Encoder().default()
    return _bm25


# ============================================
# HYBRID SEARCH FUNCTION
# ============================================

def hybrid_search(
    query:       str,
    k:           int   = 3,
    alpha:       float = 0.5,
    filter_dict: dict  = None,
) -> list[Document]:
    """
    Hybrid search: BM25 sparse + dense vectors.

    Args:
        query:       user's question
        k:           number of results to return
        alpha:       weight balance
                     1.0 = pure dense (semantic)
                     0.0 = pure sparse (keyword)
                     0.5 = equal balance ← default
        filter_dict: Pinecone metadata filter

    Returns:
        list of LangChain Document objects
        sorted by combined relevance score

    How it works:
    1. Dense:  embed query → cosine similarity
    2. Sparse: BM25 encode query → keyword match
    3. Combine: score = alpha×dense + (1-α)×sparse
    4. Return top-k by combined score
    """
    print(f"  [hybrid] query='{query[:50]}' "
          f"k={k} alpha={alpha}")

    # Step 1 — Dense vector query
    dense_vector = embeddings.embed_query(query)

    # Step 2 — Sparse BM25 vector
    bm25         = get_bm25()
    sparse_vector = bm25.encode_queries(query)

    # Step 3 — Build Pinecone hybrid query
    query_params = {
        "vector":        dense_vector,
        "sparse_vector": sparse_vector,
        "top_k":         k,
        "include_metadata": True,
        "alpha":         alpha,
        # alpha controls dense vs sparse weight
        # Pinecone hybrid search built-in parameter
    }

    if filter_dict:
        query_params["filter"] = filter_dict

    # Step 4 — Execute hybrid query
    try:
        results = index.query(**query_params)
    except Exception as e:
        # Fallback to dense-only if hybrid fails
        print(f"  [hybrid] Hybrid failed, "
              f"falling back to dense: {e}")
        query_params.pop("sparse_vector", None)
        query_params.pop("alpha", None)
        results = index.query(**query_params)

    # Step 5 — Convert to LangChain Documents
    documents = []
    for match in results.get("matches", []):
        meta    = match.get("metadata", {})
        content = meta.get("text", "")

        if not content:
            continue

        doc = Document(
            page_content = content,
            metadata     = {
                "doc_name": meta.get("doc_name", "unknown"),
                "page":     meta.get("page", 0),
                "score":    match.get("score", 0),
                "search":   "hybrid",
            }
        )
        documents.append(doc)

    print(f"  [hybrid] Found {len(documents)} results")
    return documents


def dense_search(
    query:       str,
    k:           int  = 3,
    filter_dict: dict = None,
) -> list[Document]:
    """
    Pure dense search — semantic only.
    Used when exact match not critical.
    """
    return hybrid_search(
        query, k, alpha=1.0,
        filter_dict=filter_dict
    )


def keyword_search(
    query:       str,
    k:           int  = 3,
    filter_dict: dict = None,
) -> list[Document]:
    """
    Pure keyword search — BM25 only.
    Used for exact term lookup.
    Invoice numbers, section references, codes.
    """
    return hybrid_search(
        query, k, alpha=0.0,
        filter_dict=filter_dict
    )


# ============================================
# SMART ALPHA SELECTOR
# ============================================

def smart_alpha(question: str) -> float:
    """
    Automatically choose alpha based on question type.

    Exact-term questions → lower alpha (more keyword)
    Conceptual questions → higher alpha (more semantic)

    This is where product intelligence lives.
    """
    q = question.lower()

    # Exact match signals — use more keyword search
    exact_signals = [
        "section", "clause", "article",   # legal
        "inv-", "po-", "grn-",            # invoice codes
        "gstin", "pan", "cin",            # Indian tax IDs
        "80c", "80d", "24b",              # tax sections
        "form 16", "form 26as",           # tax forms
        "schedule",                        # schedules
    ]

    # Semantic signals — use more dense search
    semantic_signals = [
        "what is", "how does", "explain",
        "why", "difference between",
        "compare", "summarise", "overview",
    ]

    exact_count    = sum(
        1 for s in exact_signals if s in q
    )
    semantic_count = sum(
        1 for s in semantic_signals if s in q
    )

    if exact_count > 0 and semantic_count == 0:
        alpha = 0.3   # mostly keyword
        print(f"  [hybrid] Exact-term query → alpha={alpha}")
    elif semantic_count > 0 and exact_count == 0:
        alpha = 0.7   # mostly semantic
        print(f"  [hybrid] Semantic query → alpha={alpha}")
    else:
        alpha = 0.5   # balanced
        print(f"  [hybrid] Balanced query → alpha={alpha}")

    return alpha