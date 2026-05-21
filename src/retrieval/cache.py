# ============================================
# src/retrieval/cache.py
# ============================================
# Semantic cache for LLM responses.
#
# Problem:
# Same question asked 100 times = 100 LLM calls
# Each call costs money and takes 3-5 seconds
#
# Solution:
# First call  → run LLM → store result
# Second call → find similar cached result → return
# No LLM call. Free. Instant.
#
# Key insight:
# "What is CRAG?" and "Tell me about CRAG"
# are semantically similar → same cache hit
# ============================================

import os
import json
import time
import numpy as np

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.92
# 0.92 = very similar questions hit cache
# Lower = more cache hits but less accurate
# Higher = fewer cache hits but more precise

class SemanticCache:
    """
    Simple in-memory semantic cache.
    Stores question embeddings + answers.
    Finds similar questions using cosine similarity.

    For production: replace dict with Redis
    For development: in-memory dict is fine
    """

    def __init__(self):
        self._embeddings = None  # lazy-loaded on first use
        self.cache     = []
        # Each entry: {
        #   "question":  original question
        #   "embedding": 384-dim vector
        #   "answer":    cached answer
        #   "sources":   cached sources
        #   "hits":      how many times served
        #   "timestamp": when cached
        # }
        self.hits   = 0
        self.misses = 0

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            print("Loading cache embeddings model...", flush=True)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
        return self._embeddings

    def _cosine_similarity(
        self, a: list, b: list
    ) -> float:
        """Cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return float(
            np.dot(a, b) /
            (np.linalg.norm(a) * np.linalg.norm(b))
        )

    def get(
        self, question: str
    ) -> dict | None:
        """
        Find cached answer for similar question.
        Returns None if no similar question found.
        """
        if not self.cache:
            self.misses += 1
            return None

        # Embed the new question
        query_embedding = self._get_embeddings().embed_query(
            question
        )

        # Find most similar cached question
        best_score = 0.0
        best_entry = None

        for entry in self.cache:
            score = self._cosine_similarity(
                query_embedding,
                entry["embedding"]
            )
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= SIMILARITY_THRESHOLD:
            best_entry["hits"] += 1
            self.hits += 1
            print(
                f"  [cache] HIT "
                f"(similarity={best_score:.3f}) "
                f"→ '{best_entry['question'][:50]}'"
            )
            return {
                "answer":  best_entry["answer"],
                "sources": best_entry["sources"],
                "cached":  True,
                "similarity": best_score,
            }

        self.misses += 1
        print(
            f"  [cache] MISS "
            f"(best={best_score:.3f})"
        )
        return None

    def set(
        self,
        question: str,
        answer:   str,
        sources:  list = None,
    ) -> None:
        """Store question + answer in cache."""
        embedding = self._get_embeddings().embed_query(question)

        self.cache.append({
            "question":  question,
            "embedding": embedding,
            "answer":    answer,
            "sources":   sources or [],
            "hits":      0,
            "timestamp": time.time(),
        })
        print(
            f"  [cache] Stored "
            f"(cache size: {len(self.cache)})"
        )

    def stats(self) -> dict:
        """Return cache performance stats."""
        total    = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size":     len(self.cache),
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": f"{hit_rate:.1%}",
            "top_cached": [
                {
                    "question": e["question"][:50],
                    "hits":     e["hits"]
                }
                for e in sorted(
                    self.cache,
                    key=lambda x: x["hits"],
                    reverse=True
                )[:5]
            ]
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache  = []
        self.hits   = 0
        self.misses = 0
        print("  [cache] Cleared")


# Singleton cache instance
# Shared across all requests
_cache = SemanticCache()

def get_cache() -> SemanticCache:
    return _cache