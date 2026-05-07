# ============================================
# DAY 4: Real PDF → Chunks → ChromaDB → Claude
# BharatRAG v0.2 — Works with any real PDF
# ============================================

import os
import shutil
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================
# STEP 1: Load real PDF
# ============================================
# PyPDFLoader reads a PDF file from disk
# Returns a list of Document objects
# Each Document = one page of the PDF
#
# Document has two parts:
# .page_content = the text on that page
# .metadata     = {"source": "file.pdf", "page": 0}
#
# Compare to Day 3:
# Day 3: you typed sentences manually
# Day 4: loader reads any PDF automatically
# ============================================

PDF_PATH = "./SmartDocs_Complete_Learning_Guide.pdf"

print(f"Loading PDF: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Loaded {len(pages)} pages")
print(f"First page preview: {pages[0].page_content[:200]}")
print()

# ============================================
# STEP 2: Split into chunks
# ============================================
# RecursiveCharacterTextSplitter is the best
# default splitter. It tries to split on:
# 1. Paragraphs (\n\n) first
# 2. Sentences (\n) if paragraphs too long
# 3. Words (space) if sentences too long
# 4. Characters as last resort
#
# chunk_size=1000   = max characters per chunk
# chunk_overlap=200 = shared characters between
#                     adjacent chunks
#
# WHY OVERLAP MATTERS:
# Without overlap:
# Chunk 1: "...GST penalty is Rs 50"
# Chunk 2: "per day for late filing..."
# → Question about "penalty per day" finds nothing!
#
# With overlap:
# Chunk 1: "...GST penalty is Rs 50"
# Chunk 2: "Rs 50 per day for late filing..."
# → Both chunks found for same question ✅
#
# Compare to Day 3:
# Day 3: chunks were manually written sentences
# Day 4: splitter creates chunks automatically
# ============================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = splitter.split_documents(pages)

print(f"Split into {len(chunks)} chunks")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print()

# Show what a chunk looks like
print("Sample chunk:")
print(f"Content: {chunks[0].page_content[:300]}")
print(f"Metadata: {chunks[0].metadata}")
print()

# ============================================
# STEP 3: Create embeddings + ChromaDB
# ============================================
# Same as Day 3 but now storing REAL chunks
# not hardcoded sentences
# ============================================

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Always delete old DB to avoid duplicates
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("Cleared old ChromaDB")

# Store all chunks in ChromaDB
print(f"Storing {len(chunks)} chunks in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"Stored {vectorstore._collection.count()} chunks")
print()

# ============================================
# STEP 4: Create retriever
# k=3 now — more chunks for richer context
# ============================================
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ============================================
# STEP 5: Claude LLM
# ============================================
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ============================================
# STEP 6: Prompt template
# ============================================
template = """You are a helpful document assistant.

Use ONLY the following context to answer.
If the answer is not in the context say:
"I could not find this information in the document."

Always cite which page your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# ============================================
# STEP 7: Format docs with page citations
# ============================================
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        # Page is 0-indexed — add 1 for human reading
        formatted.append(
            f"[Page {int(page)+1}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

# ============================================
# STEP 8: Build RAG chain (same as Day 3)
# ============================================
rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================
# STEP 9: Test with real questions about
# the SmartDocs learning guide PDF
# ============================================
print("=" * 60)
print("BharatRAG v0.2 — Real PDF loaded")
print(f"Document: {PDF_PATH}")
print(f"Chunks: {vectorstore._collection.count()}")
print("=" * 60)
print()

questions = [
    "What is SmartDocs AI?",
    "What is the difference between V1 and V2?",
    "What is session state in Streamlit?",
    "What is context stuffing?",
    "How much did SmartDocs cost to build?",
]

for question in questions:
    print(f"Q: {question}")
    print("-" * 40)
    answer = rag_chain.invoke(question)
    print(f"A: {answer}")
    print()

# ============================================
# INTERACTIVE MODE
# ============================================
print("=" * 60)
print("Ask your own questions about the document")
print("Type 'exit' to quit")
print("=" * 60)

while True:
    user_input = input("\nYour question: ").strip()
    if user_input.lower() in ["exit", "quit", ""]:
        break
    answer = rag_chain.invoke(user_input)
    print(f"\nAnswer: {answer}")