# ============================================
# test_hybrid.py — Day 20 hybrid search test
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()

from src.retrieval.hybrid import (
    hybrid_search, dense_search,
    keyword_search, smart_alpha
)

def test_comparison(query: str):
    print(f"\nQuery: '{query}'")
    print("-" * 50)

    # Dense only
    print("Dense (semantic):")
    dense = dense_search(query, k=3)
    for d in dense:
        print(f"  [{d.metadata.get('doc_name','?')} "
              f"p.{int(d.metadata.get('page',0))+1}] "
              f"{d.page_content[:60]}...")

    # Hybrid
    print("\nHybrid (balanced):")
    hybrid = hybrid_search(query, k=3, alpha=0.5)
    for d in hybrid:
        print(f"  [{d.metadata.get('doc_name','?')} "
              f"p.{int(d.metadata.get('page',0))+1}] "
              f"{d.page_content[:60]}...")

    # Smart alpha
    alpha = smart_alpha(query)
    print(f"\nSmart alpha selected: {alpha}")


if __name__ == "__main__":
    print("=" * 55)
    print("Day 20: Hybrid Search Tests")
    print("=" * 55)

    # Test 1 — semantic query
    # Dense should do well here
    test_comparison("What is session state in Streamlit?")

    # Test 2 — exact term query
    # Hybrid should do better than dense alone
    test_comparison("What is CRAG Section 80C approach?")

    # Test 3 — invoice query
    # Keyword should dominate
    test_comparison("total amount invoice GST CGST SGST")