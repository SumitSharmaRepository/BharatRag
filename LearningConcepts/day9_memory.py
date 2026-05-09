# ============================================
# DAY 9: Conversation Memory + Multi-Document
# ============================================
# DAY 8 RECAP:
# Single question agent with hallucination check
# Each question treated independently
# No memory of previous conversation
#
# DAY 9 ADDS:
# 1. chat_history in state — remembers conversation
# 2. generate_node sends history to Claude
# 3. Multiple PDFs loaded simultaneously
# 4. Each chunk tagged with source document
#
# WHY MEMORY MATTERS:
# Without: "What is V2?" → "What about V3?"
#          Agent doesn't know what "it" refers to
#
# With:    "What is V2?" → "What about V3?"
#          Agent knows V3 = V3 of SmartDocs
#          because V2 question is in history
# ============================================

import os
import glob
import shutil
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Setup ─────────────────────────────────────────────
CHROMA_PATH     = "/home/sumit/bharatrag/chroma_db"
DATA_PATH       = "/home/sumit/bharatrag/data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")

# ============================================
# MULTI-DOCUMENT LOADER
# ============================================
# Day 4-8: one PDF hardcoded in path
# Day 9: loads ALL PDFs from data/ folder
#
# Each chunk gets metadata:
# {"source": "filename.pdf", "page": 3}
# So answers can cite WHICH document
# ============================================

def load_all_pdfs(data_path: str) -> list:
    """
    Load all PDFs from data folder.
    Returns list of chunks with source metadata.
    """
    pdf_files = glob.glob(
        os.path.join(data_path, "*.pdf")
    )

    if not pdf_files:
        print(f"No PDFs found in {data_path}")
        return []

    print(f"Found {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"  → {os.path.basename(f)}")

    all_chunks = []
    splitter   = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 100,
    )

    for pdf_path in pdf_files:
        print(f"\nLoading: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()

        # Add document name to metadata
        # So we know WHICH PDF each chunk came from
        for page in pages:
            page.metadata["doc_name"] = os.path.basename(
                pdf_path
            )

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"  {len(pages)} pages → {len(chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks across "
          f"{len(pdf_files)} document(s)")
    return all_chunks


def setup_vectorstore(force_reload: bool = False):
    """
    Load or create ChromaDB with all PDFs.
    force_reload=True: re-index even if DB exists
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    if os.path.exists(CHROMA_PATH) and not force_reload:
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
        print(f"Loaded {vectorstore._collection.count()} chunks")
    else:
        print("Indexing PDFs into ChromaDB...")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        chunks = load_all_pdfs(DATA_PATH)
        if not chunks:
            raise ValueError("No PDF chunks to index")

        vectorstore = Chroma.from_documents(
            documents         = chunks,
            embedding         = embeddings,
            persist_directory = CHROMA_PATH,
        )
        print(f"Indexed {vectorstore._collection.count()} chunks")

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

retriever = setup_vectorstore()

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    anthropic_api_key=ANTHROPIC_KEY
)

# ============================================
# STATE — New field: chat_history
# ============================================

class RAGState(TypedDict):
    question:            str
    chat_history:        List[dict]  # NEW
    rewritten_question:  str
    documents:           List[str]
    answer:              str
    grade:               str
    hallucination:       str
    attempts:            int
    generation_attempts: int

# ============================================
# NODES
# ============================================

def retrieve_node(state: RAGState) -> dict:
    """
    Retrieve from ChromaDB.
    Uses rewritten query if available.
    Same as Day 8.
    """
    attempts  = state.get("attempts", 0)
    rewritten = state.get("rewritten_question", "")

    if attempts > 0 and rewritten:
        query = rewritten
        print(f"  [retrieve] Rewritten: '{query}'")
    else:
        query = state["question"]
        print(f"  [retrieve] Original: '{query}'")

    docs      = retriever.invoke(query)
    doc_texts = []

    for i, doc in enumerate(docs):
        page     = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get(
            "doc_name", "unknown.pdf"
        )
        print(f"    Chunk {i+1} "
              f"[{doc_name}, p.{int(page)+1}]: "
              f"{doc.page_content[:50]}...")
        doc_texts.append(
            f"[{doc_name}, Page {int(page)+1}]\n"
            f"{doc.page_content}"
        )
        # Include doc_name in text so generate_node
        # can cite the source in its answer

    return {
        "documents": doc_texts,
        "attempts":  attempts + 1
    }


def grade_node(state: RAGState) -> dict:
    """Grade relevance. Same as Day 8."""
    question  = state["question"]
    documents = state["documents"]
    print(f"  [grade] Checking relevance...")

    prompt = f"""Are these documents relevant to: "{question}"?

Documents:
{chr(10).join(documents)}

Reply ONLY: relevant or irrelevant"""

    response = llm.invoke([HumanMessage(content=prompt)])
    grade    = response.content.strip().lower()
    result   = "irrelevant" if "irrelevant" in grade else "relevant"

    print(f"  [grade] Result: {result}")
    return {"grade": result}


def query_rewrite_node(state: RAGState) -> dict:
    """Rewrite query on failure. Same as Day 8."""
    original  = state["question"]
    documents = state["documents"]
    print(f"  [rewrite] Rewriting failed query...")

    prompt = f"""Question failed to find relevant docs: "{original}"

Retrieved (irrelevant):
{chr(10).join(documents[:2])}

Rewrite as specific technical question.
Under 15 words. Just the question.

Rewritten:"""

    response  = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip().strip('"').strip("'")
    print(f"  [rewrite] New: '{rewritten}'")
    return {"rewritten_question": rewritten}


def generate_node(state: RAGState) -> dict:
    """
    Generate answer — NOW WITH MEMORY.

    Key Day 9 change:
    Sends chat_history to Claude alongside context.
    Claude understands follow-up questions because
    it can see the previous conversation.

    Example:
    History: [user: "What is V2?", assistant: "V2 adds..."]
    Current: "What about V3?"
    Claude now knows V3 = V3 of SmartDocs AI
    """
    question         = state["question"]
    documents        = state["documents"]
    chat_history     = state.get("chat_history", [])
    gen_attempts     = state.get("generation_attempts", 0)

    print(f"  [generate] Creating answer "
          f"(attempt {gen_attempts + 1}, "
          f"history: {len(chat_history)} messages)...")

    context = "\n\n".join(documents)

    # ── Format chat history for Claude ──────────────
    # Convert list of dicts to readable conversation
    # Claude uses this to understand follow-up questions
    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in chat_history[-4:]:
            # Only last 4 messages to avoid huge context
            # Older history matters less
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role.upper()}: {content}\n"

    if gen_attempts == 0:
        prompt = f"""You are a helpful document assistant.

Answer using ONLY the context below.
If not found say: "I could not find this in the document."
Always cite which document your answer comes from.
{history_text}
Context:
{context}

Current question: {question}

Answer:"""
    else:
        # Strict prompt on regeneration
        prompt = f"""STRICT: Answer using ONLY exact 
information from context. No outside knowledge.
{history_text}
Context:
{context}

Question: {question}

Strictly grounded answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "answer":             response.content,
        "generation_attempts": gen_attempts + 1
    }


def hallucination_check_node(state: RAGState) -> dict:
    """LLM-as-judge. Same as Day 8."""
    answer    = state["answer"]
    documents = state["documents"]
    question  = state["question"]
    print(f"  [hallucination_check] Verifying...")

    prompt = f"""Fact-check this answer against context.

Context:
{chr(10).join(documents)}

Question: {question}
Answer: {answer}

Is every claim supported by context?
Reply ONLY: grounded or hallucinated"""

    response      = llm.invoke([HumanMessage(content=prompt)])
    result        = response.content.strip().lower()
    hallucination = "hallucinated" if "hallucinated" in result \
                    else "grounded"

    print(f"  [hallucination_check] Result: {hallucination}")
    return {"hallucination": hallucination}


def memory_node(state: RAGState) -> dict:
    """
    Node (NEW): Save Q&A to chat_history.

    Runs AFTER successful answer.
    Appends current question and answer to history.
    Next question will have this context available.

    This is what makes conversation feel natural.
    Without this: every question is isolated.
    With this: questions build on each other.
    """
    question     = state["question"]
    answer       = state["answer"]
    chat_history = state.get("chat_history", [])

    print(f"  [memory] Saving to history "
          f"(total: {len(chat_history) + 2} messages)")

    # Append current Q&A to history
    updated_history = chat_history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    return {"chat_history": updated_history}


def fallback_node(state: RAGState) -> dict:
    """Honest fallback. Same as Day 8."""
    original  = state["question"]
    rewritten = state.get("rewritten_question", "")
    print(f"  [fallback] All attempts failed.")

    if rewritten:
        answer = (
            f"I searched for '{original}' and "
            f"'{rewritten}' but found no relevant "
            f"information in the documents."
        )
    else:
        answer = (
            "I could not find relevant information "
            "to answer your question."
        )
    return {"answer": answer}


# ============================================
# ROUTERS
# ============================================

def should_generate(state: RAGState) -> str:
    grade    = state.get("grade", "irrelevant")
    attempts = int(state.get("attempts", 0))
    print(f"  [router] grade={grade} attempts={attempts}")

    if grade == "relevant":
        print("  [router] → generate")
        return "generate"
    elif attempts == 1:
        print("  [router] → rewrite")
        return "rewrite"
    else:
        print("  [router] → fallback")
        return "fallback"


def should_return(state: RAGState) -> str:
    hallucination = state.get("hallucination", "grounded")
    gen_attempts  = int(state.get("generation_attempts", 0))
    print(f"  [router] hallucination={hallucination} "
          f"gen_attempts={gen_attempts}")

    if hallucination == "grounded":
        print("  [router] → save to memory")
        return "save_memory"
    elif gen_attempts < 2:
        print("  [router] → regenerate")
        return "regenerate"
    else:
        print("  [router] → save anyway")
        return "save_memory"


# ============================================
# BUILD GRAPH
# ============================================
# Day 8 graph + memory_node added after hallucination
#
# New flow after generate:
# generate → hallucination_check → should_return
#                                → grounded    → memory_node → END
#                                → hallucinated → generate (retry)
# ============================================

def build_agent():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve",            retrieve_node)
    workflow.add_node("grade",               grade_node)
    workflow.add_node("rewrite",             query_rewrite_node)
    workflow.add_node("generate",            generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("memory",              memory_node)  # NEW
    workflow.add_node("fallback",            fallback_node)

    # Fixed edges
    workflow.add_edge("retrieve",            "grade")
    workflow.add_edge("rewrite",             "retrieve")
    workflow.add_edge("generate",            "hallucination_check")
    workflow.add_edge("memory",              END)   # NEW
    workflow.add_edge("fallback",            END)

    # Conditional edges
    workflow.add_conditional_edges(
        "grade",
        should_generate,
        {
            "generate": "generate",
            "rewrite":  "rewrite",
            "fallback": "fallback",
        }
    )

    workflow.add_conditional_edges(
        "hallucination_check",
        should_return,
        {
            "save_memory": "memory",    # grounded → save
            "regenerate":  "generate",  # hallucinated → retry
        }
    )

    workflow.set_entry_point("retrieve")
    return workflow.compile()


# ============================================
# CONVERSATIONAL RUNNER
# ============================================

def run_conversation(agent):
    """
    Run a multi-turn conversation.
    chat_history persists across questions.
    Each question builds on previous context.
    """
    print("=" * 55)
    print("BharatRAG v0.4 — Conversational Agent")
    print("Type 'exit' to quit | 'history' to see chat log")
    print("=" * 55)
    print()

    # Persistent history across all questions
    chat_history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if question.lower() == "history":
            print("\n── Chat History ──")
            for msg in chat_history:
                role    = msg["role"].upper()
                content = msg["content"][:100]
                print(f"{role}: {content}...")
            print()
            continue

        print()
        initial_state = {
            "question":            question,
            "chat_history":        chat_history,  # pass history
            "rewritten_question":  "",
            "documents":           [],
            "answer":              "",
            "grade":               "irrelevant",
            "hallucination":       "grounded",
            "attempts":            0,
            "generation_attempts": 0,
        }

        final_state  = agent.invoke(initial_state)
        answer       = final_state["answer"]
        chat_history = final_state["chat_history"]
        # Update history from final state
        # memory_node appended current Q&A to it

        print(f"\nBharatRAG: {answer}")
        print(f"[History: {len(chat_history)} messages]\n")


# ============================================
# DEMO MODE — shows memory working
# ============================================

def run_demo(agent):
    """
    Run preset questions to demonstrate memory.
    Questions build on each other deliberately.
    """
    print("=" * 55)
    print("DEMO MODE — Showing conversation memory")
    print("=" * 55)

    chat_history = []

    demo_questions = [
        "What is SmartDocs AI?",
        "What is the difference between V1 and V2?",
        "Which version added error handling?",   # needs history
        "What about V3 — what did it add?",      # needs history
        "How to make biryani?",                  # should fallback
    ]

    for question in demo_questions:
        print(f"\nYou: {question}")
        print("-" * 40)

        initial_state = {
            "question":            question,
            "chat_history":        chat_history,
            "rewritten_question":  "",
            "documents":           [],
            "answer":              "",
            "grade":               "irrelevant",
            "hallucination":       "grounded",
            "attempts":            0,
            "generation_attempts": 0,
        }

        final_state  = agent.invoke(initial_state)
        answer       = final_state["answer"]
        chat_history = final_state["chat_history"]

        print(f"\nBharatRAG: {answer}")
        print(f"[History: {len(chat_history)} messages]")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("Building Day 9 conversational agent...")
    agent = build_agent()
    print("Agent ready!")
    print()

    print("Choose mode:")
    print("1. Demo (preset questions showing memory)")
    print("2. Interactive (type your own questions)")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        run_demo(agent)
    else:
        run_conversation(agent)