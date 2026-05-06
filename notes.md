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