# ============================================
# DAY 13: Migrate ChromaDB → Pinecone
# ============================================
# Why Pinecone over ChromaDB:
# ChromaDB = local disk, dies on deployment
# Pinecone  = cloud, survives anywhere
#
# Why Pinecone over Supabase:
# Pinecone = purpose built for vectors
#            zero version conflicts
#            5 lines to connect
#            most mentioned in job postings
# ============================================

import os
import glob
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, "/home/sumit/bharatrag")

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings import get_embeddings

# ── Credentials ───────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "bharatrag")
DATA_PATH        = "/home/sumit/bharatrag/data"

# ── Connect to Pinecone ───────────────────────────────
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
print("Connected!")

# ── Embedding model ───────────────────────────────────
print("Using Pinecone inference embeddings (multilingual-e5-large, 1024 dims)...")
embeddings = get_embeddings()
print("Embeddings ready")

# ============================================
# STEP 1: Load all PDFs
# Same as always — nothing changes here
# ============================================

def load_all_pdfs() -> list:
    pdf_files  = glob.glob(f"{DATA_PATH}/*.pdf")
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 100,
    )

    print(f"\nFound {len(pdf_files)} PDF(s):")
    for pdf_path in pdf_files:
        name   = os.path.basename(pdf_path)
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()

        for page in pages:
            page.metadata["doc_name"] = name

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"  {name}: {len(pages)} pages "
              f"→ {len(chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks")
    return all_chunks


# ============================================
# STEP 2: Store in Pinecone
# ============================================
# PineconeVectorStore.from_documents():
# 1. Takes each chunk
# 2. Embeds it → 384 numbers
# 3. Stores in Pinecone index with metadata
#
# Identical interface to:
# Chroma.from_documents()       ← Day 4-5
# SupabaseVectorStore.from_documents() ← attempted Day 13
#
# Same LangChain interface. Different backend.
# THIS is why LangChain is powerful.
# ============================================

def migrate_to_pinecone(chunks: list) -> PineconeVectorStore:
    """
    Store all chunks in Pinecone.
    Returns vectorstore for immediate use.
    """
    print(f"\nMigrating {len(chunks)} chunks to Pinecone...")
    print("Processing in batches of 100...")

    # Pinecone handles batching internally
    # but we process in groups to show progress
    BATCH_SIZE = 100
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch      = chunks[i:i + BATCH_SIZE]
        batch_num  = i // BATCH_SIZE + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) \
                        // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches}: "
              f"chunks {i+1}-"
              f"{min(i+BATCH_SIZE, len(chunks))}...")

        if vectorstore is None:
            # First batch — create vectorstore
            vectorstore = PineconeVectorStore.from_documents(
                documents = batch,
                embedding = embeddings,
                index_name = PINECONE_INDEX,
            )
        else:
            # Subsequent batches — add to existing
            vectorstore.add_documents(batch)

    print(f"\n✅ All {len(chunks)} chunks stored in Pinecone!")
    return vectorstore


# ============================================
# STEP 3: Verify migration
# ============================================

def verify_migration() -> PineconeVectorStore:
    """
    Load existing Pinecone index and test search.
    """
    print("\nVerifying migration...")

    # Connect to existing index
    # from_documents = creates new
    # PineconeVectorStore = loads existing
    vectorstore = PineconeVectorStore(
        index_name = PINECONE_INDEX,
        embedding  = embeddings,
    )

    # Check vector count via Pinecone stats
    index       = pc.Index(PINECONE_INDEX)
    stats       = index.describe_index_stats()
    total       = stats.get(
        "total_vector_count", "unknown"
    )
    print(f"Vectors in Pinecone: {total}")

    # Test semantic search
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    test_queries = [
        "What is session state in Streamlit?",
        "What is CRAG?",
        "What is the difference between V1 and V2?",
    ]

    print("\nTest retrieval results:")
    print("-" * 50)

    for query in test_queries:
        docs = retriever.invoke(query)
        print(f"\nQuery: '{query}'")
        for i, doc in enumerate(docs):
            name = doc.metadata.get("doc_name", "?")
            page = doc.metadata.get("page", "?")
            print(f"  [{i+1}] {name} "
                  f"p.{int(page)+1}: "
                  f"{doc.page_content[:60]}...")

    print("\n✅ Verification complete!")
    return vectorstore


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 55)
    print("BharatRAG — ChromaDB → Pinecone Migration")
    print("=" * 55)

    # Step 1: Load PDFs
    chunks = load_all_pdfs()

    # Step 2: Migrate
    migrate_to_pinecone(chunks)

    # Step 3: Verify
    verify_migration()

    print("\n" + "=" * 55)
    print("Day 13 complete!")
    print(f"Chunks stored in Pinecone: {PINECONE_INDEX}")
    print("Next: update api.py to use Pinecone")
    print("=" * 55)