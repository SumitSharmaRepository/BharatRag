# ============================================
# src/specialists/logistics_agent.py
# ============================================
# NEW in Day 18.
# Handles invoice PDFs, delivery challans,
# purchase orders, e-way bills.
#
# Indian logistics documents:
# → GST invoices (tax amounts, GSTIN)
# → Delivery challans (quantity, weight)
# → Purchase orders (rates, terms)
# → E-way bills (vehicle, route)
# ============================================

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from src.agents.state import MultiAgentState

ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "bharatrag")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

llm = ChatAnthropic(
    model             = "claude-sonnet-4-5",
    temperature       = 0,
    anthropic_api_key = ANTHROPIC_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# Language instructions
LANG_INSTRUCTIONS = {
    "English":       "Answer in clear English.",
    "Hindi / हिंदी": "हिंदी में जवाब दें।",
    "Hinglish":      "Answer in Hinglish naturally.",
    "Arabic / عربي": "أجب باللغة العربية بوضوح.",
}


def get_logistics_retriever():
    """
    Retriever filtered to logistics documents only.
    Searches: invoices, challans, purchase orders.
    """
    pc          = Pinecone(api_key=PINECONE_API_KEY)
    vectorstore = PineconeVectorStore(
        index_name = PINECONE_INDEX,
        embedding  = embeddings,
    )
    return vectorstore.as_retriever(
        search_kwargs={
            "k":      4,
            "filter": {
                "doc_type": {"$in": [
                    "invoice",
                    "challan",
                    "purchase_order",
                    "eway_bill",
                    "logistics",
                ]}
            }
        }
    )


def logistics_agent_node(
    state: MultiAgentState
) -> dict:
    """
    Specialist agent for logistics documents.

    Optimised for:
    → Extracting amounts and totals
    → Finding dates and deadlines
    → Comparing vendors and rates
    → Indian number formatting (lakhs, crores)
    → GST calculations on invoices

    Args:
        state: MultiAgentState with question,
               user_facts, language, user_id

    Returns:
        dict with answer and agent_used
    """
    question   = state["question"]
    user_facts = state.get("user_facts", "")
    language   = state.get("language", "English")

    print(f"  [LogisticsAgent] Handling: '{question}'")

    # Retrieve logistics chunks
    try:
        retriever = get_logistics_retriever()
        docs = hybrid_search(
            query  = question,
            k      = 4,
            alpha  = 0.3,  # more keyword for invoices
        )
    except Exception:
        # Fallback: search without filter
        # if no logistics docs indexed yet
        pc          = Pinecone(api_key=PINECONE_API_KEY)
        vectorstore = PineconeVectorStore(
            index_name = PINECONE_INDEX,
            embedding  = embeddings,
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        docs = hybrid_search(
            query  = question,
            k      = 4,
            alpha  = 0.3,  # more keyword for invoices
        )

    if not docs:
        return {
            "answer":     "No logistics documents found. "
                         "Please upload invoice or "
                         "delivery challan PDFs first.",
            "agent_used": "LogisticsAgent",
            "documents":  [],
        }

    # Format with citations
    doc_texts = []
    for doc in docs:
        page     = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get("doc_name", "document")
        doc_texts.append(
            f"[{doc_name}, Page {int(page)+1}]\n"
            f"{doc.page_content}"
        )

    context           = "\n\n".join(doc_texts)
    lang_instruction  = LANG_INSTRUCTIONS.get(
        language, "Answer in clear English."
    )

    # Memory-aware logistics prompt
    memory_section = ""
    if user_facts:
        memory_section = f"\n{user_facts}\n"

    prompt = f"""You are a logistics and invoice \
analysis specialist for Indian businesses.
{memory_section}
{lang_instruction}

Specialise in:
- Extracting exact amounts, totals, tax values
- Identifying dates (invoice date, due date, delivery date)
- Comparing rates across vendors
- Calculating GST breakdowns (CGST, SGST, IGST)
- Using Indian number format (lakhs, crores)
- Citing invoice numbers and document names

Answer using ONLY the provided context.
If not found: "This information is not in the \
uploaded logistics documents."
Always cite document name and page number.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"  [LogisticsAgent] Answer ready")

    return {
        "answer":     response.content,
        "agent_used": "LogisticsAgent",
        "documents":  doc_texts,
    }