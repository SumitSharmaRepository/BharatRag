DAY 1 — Manual embeddings:
model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = model.encode(sentences)
← YOU did the embedding manually

DAY 2 — ChromaDB automatic embeddings:
collection.add(documents=documents, ids=ids)
← ChromaDB embedded internally
← You never saw it happen
← ChromaDB downloaded its own ONNX version
   of the same model (that 79MB download)

DAY 3 — LangChain embeddings:
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(embedding_function=embeddings)
← LangChain uses embeddings to SEARCH ChromaDB
← Must be SAME model as Day 2
← Otherwise vectors won't match


requirements.txt  → list of packages needed
                    anyone can install with:
                    pip install -r requirements.txt
                    Like pom.xml in Maven

__init__.py       → marks folder as Python package
                    enables: from src.loader import ...
                    can contain convenience imports
                    empty = valid, just marks package
                    with imports = cleaner main.py

Python has no compiler — it's an interpreter.
The correct term is: tells the Python INTERPRETER
that this folder is a PACKAGE.

Enables:
from src.loader import load_and_chunk_pdf
          ↑
      This works ONLY because src/__init__.py exists

create_vectorstore():
→ Takes chunks as input
→ Embeds them (slow — runs embedding model)
→ Stores vectors to disk
→ Use when: new PDF uploaded, first time indexing

load_vectorstore():
→ No chunks needed
→ Reads existing vectors from disk (fast)
→ Does NOT re-embed anything
→ Use when: PDF already indexed, just want to query


load_pdf()             → loads PDF, returns pages only
chunk_documents()      → splits pages into chunks
load_and_chunk_pdf()   → calls BOTH in sequence
                         convenience wrapper
                         one call instead of two