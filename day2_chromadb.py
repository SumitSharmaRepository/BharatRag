import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="tax_documents"
)

documents = [
    "GST return filing deadline is the 20th of every month.",
    "Late filing of GST attracts penalty of Rs 50 per day.",
    "Income tax return must be filed by July 31st.",
    "TDS must be deposited by 7th of following month.",
    "Section 80C allows deduction up to Rs 1.5 lakh.",
    "How to make biryani at home with basmati rice.",
    "Best cricket matches of 2025 highlights.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

metadatas = [
    {"source": "GST_guide.pdf",  "page": 1, "topic": "GST"},
    {"source": "GST_guide.pdf",  "page": 2, "topic": "GST"},
    {"source": "ITR_guide.pdf",  "page": 1, "topic": "Income Tax"},
    {"source": "TDS_guide.pdf",  "page": 1, "topic": "TDS"},
    {"source": "ITR_guide.pdf",  "page": 5, "topic": "Deductions"},
    {"source": "random.pdf",     "page": 1, "topic": "Food"},
    {"source": "random.pdf",     "page": 2, "topic": "Sports"},
]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"Stored {collection.count()} chunks in ChromaDB")
print()

queries = [
    "When is GST filing due?",
    "What deductions can I claim?",
    "How to cook rice?",
]

for query in queries:
    print(f"Query: '{query}'")
    print("-" * 50)

    results = collection.query(
        query_texts=[query],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = 1 - (dist / 2)
        print(f"  Score: {similarity:.3f}")
        print(f"  Source: {meta['source']}, Page: {meta['page']}")
        print(f"  Text: {doc[:70]}...")
        print()
    print()