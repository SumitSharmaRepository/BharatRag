# ============================================
# src/embeddings.py
# ============================================
# Pinecone hosted inference embeddings.
# Replaces HuggingFaceEmbeddings everywhere.
#
# Why: sentence-transformers downloads PyTorch
#      at runtime → 3-4 min Render startup.
#      Pinecone inference = API call, no model
#      download, <1s startup.
#
# Model: multilingual-e5-large (1024 dims)
# Supports 100+ languages including Hindi.
# ============================================

import os
from typing import List
from langchain_core.embeddings import Embeddings
from pinecone import Pinecone

PINECONE_EMBED_MODEL = "multilingual-e5-large"
EMBED_BATCH_SIZE     = 96   # Pinecone inference max inputs


class PineconeEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings using Pinecone's
    hosted inference API (multilingual-e5-large, 1024 dims).

    Implements embed_documents() and embed_query() so it
    works as a drop-in replacement for HuggingFaceEmbeddings
    in LangChain, PineconeVectorStore, and Mem0.
    """

    def __init__(self):
        self._pc = None

    def _client(self) -> Pinecone:
        if self._pc is None:
            self._pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        return self._pc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (passages) in batches."""
        pc      = self._client()
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch  = texts[i : i + EMBED_BATCH_SIZE]
            result = pc.inference.embed(
                model      = PINECONE_EMBED_MODEL,
                inputs     = batch,
                parameters = {"input_type": "passage"},
            )
            vectors.extend(item.values for item in result)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query."""
        pc     = self._client()
        result = pc.inference.embed(
            model      = PINECONE_EMBED_MODEL,
            inputs     = [text],
            parameters = {"input_type": "query"},
        )
        return result[0].values


# Singleton — one instance shared across all modules
_embeddings: PineconeEmbeddings = None


def get_embeddings() -> PineconeEmbeddings:
    """Return the shared PineconeEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = PineconeEmbeddings()
    return _embeddings
