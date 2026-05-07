# ============================================
# src/vectorstore.py — ChromaDB management
# Single responsibility: store and search vectors
# ============================================

import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, CHROMA_DB_PATH, RETRIEVAL_K

def get_embeddings():
    """
    Create and return the embedding model.
    Cached pattern — create once, reuse many times.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

def create_vectorstore(chunks: list,
                       reset: bool = True) -> Chroma:
    """
    Store document chunks in ChromaDB.

    Args:
        chunks: list of Document chunks from loader
        reset:  if True, delete old DB before creating
                prevents duplicate chunks from reruns

    Returns:
        Chroma vectorstore object
    """
    embeddings = get_embeddings()

    if reset and os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print("Cleared existing ChromaDB")

    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        persist_directory = CHROMA_DB_PATH,
    )
    print(f"Stored {vectorstore._collection.count()} chunks")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Load existing ChromaDB without re-indexing.
    Use this when PDF is already indexed.

    Returns:
        Existing Chroma vectorstore
    """
    embeddings = get_embeddings()
    return Chroma(
        embedding_function = embeddings,
        persist_directory  = CHROMA_DB_PATH,
    )


def get_retriever(vectorstore: Chroma):
    """
    Create retriever from vectorstore.

    k=RETRIEVAL_K means return top k most similar chunks.
    Configured in config.py.
    """
    return vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )