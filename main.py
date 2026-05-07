# ============================================
# main.py — Entry point for BharatRAG
# Ties all modules together
# ============================================

import os
from src.loader      import load_and_chunk_pdf
from src.vectorstore import create_vectorstore, get_retriever
from src.chain       import create_rag_chain

def main():
    print("=" * 60)
    print("BharatRAG v0.3 — Modular Architecture")
    print("=" * 60)
    print()

    # ── Step 1: Load PDF ──────────────────────────────
    pdf_path = "./data/SmartDocs_Complete_Learning_Guide.pdf"

    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        print("Please copy your PDF to the data/ folder")
        return

    chunks = load_and_chunk_pdf(pdf_path)
    print()

    # ── Step 2: Store in ChromaDB ─────────────────────
    vectorstore = create_vectorstore(chunks, reset=True)
    retriever   = get_retriever(vectorstore)
    print()

    # ── Step 3: Create RAG chain ──────────────────────
    chain = create_rag_chain(retriever, language="English")
    print()

    # ── Step 4: Test questions ────────────────────────
    print("Testing BharatRAG v0.3...")
    print("-" * 40)

    questions = [
        "What is SmartDocs AI?",
        "What is the difference between V1 and V2?",
        "What is session state in Streamlit?",
    ]

    for question in questions:
        print(f"Q: {question}")
        answer = chain.invoke(question)
        print(f"A: {answer}")
        print()

    # ── Step 5: Interactive mode ──────────────────────
    print("=" * 60)
    print("Interactive mode — type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ["exit", "quit", ""]:
            break
        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()