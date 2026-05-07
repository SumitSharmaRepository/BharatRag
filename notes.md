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