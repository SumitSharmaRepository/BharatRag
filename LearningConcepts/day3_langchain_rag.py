import os
import shutil
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================
# STEP 1: Create embeddings model
# Same model used for storing AND searching
# ============================================
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================
# STEP 2: Create/Load ChromaDB with documents
# ============================================
documents_text = [
    "GST return filing deadline is the 20th of every month.",
    "Late filing of GST attracts penalty of Rs 50 per day.",
    "Income tax return must be filed by July 31st.",
    "TDS must be deposited by 7th of following month.",
    "Section 80C allows deduction up to Rs 1.5 lakh.",
    "How to make biryani at home with basmati rice.",
    "Best cricket matches of 2025 highlights.",
]

metadatas = [
    {"source": "GST_guide.pdf",  "page": 1},
    {"source": "GST_guide.pdf",  "page": 2},
    {"source": "ITR_guide.pdf",  "page": 1},
    {"source": "TDS_guide.pdf",  "page": 1},
    {"source": "ITR_guide.pdf",  "page": 5},
    {"source": "random.pdf",     "page": 1},
    {"source": "random.pdf",     "page": 2},
]

print("Creating ChromaDB with LangChain embeddings...")
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
vectorstore = Chroma.from_texts(
    texts=documents_text,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./chroma_db"
)
print(f"Stored {vectorstore._collection.count()} chunks")
print()

# ============================================
# STEP 3: Create retriever
# k=2 returns top 2 most relevant chunks
# ============================================
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# ============================================
# STEP 4: Create Claude LLM
# ============================================
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ============================================
# STEP 5: Prompt Template
# ============================================
template = """You are a helpful document assistant
for Indian tax and financial documents.

Use ONLY the following context to answer.
If answer not in context say:
"I could not find this information in the documents."

Always mention which document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# ============================================
# STEP 6: Format retrieved docs with citations
# ============================================
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

# ============================================
# STEP 7: Build RAG Chain
# | means pipe — output flows to next step
# ============================================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================
# STEP 8: Test with questions
# ============================================
print("=" * 60)
print("BharatRAG v0.1 — ChromaDB + LangChain + Claude")
print("=" * 60)
print()

questions = [
    "When is the GST filing deadline?",
    "What is the penalty for late GST filing?",
    "What deductions can I claim?",
    "How to make biryani?",
]

for question in questions:
    print(f"Q: {question}")
    print("-" * 40)
    answer = rag_chain.invoke(question)
    print(f"A: {answer}")
    print()