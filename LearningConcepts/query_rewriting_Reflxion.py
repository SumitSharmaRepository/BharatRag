# ============================================
# DAY 7: Query Rewriting — Reflexion Pattern
# ============================================
# DAY 6 RECAP:
# retrieve → grade → irrelevant → retry SAME query
# Problem: same query = same bad chunks always
#
# DAY 7 ADDS:
# retrieve → grade → irrelevant → REWRITE query
#                              → retrieve NEW query
#                              → better results
#
# This is called REFLEXION in agentic AI:
# Agent reflects on WHY it failed
# Changes its approach before retrying
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# ── ChromaDB setup ────────────────────────────────────
CHROMA_PATH     = "/home/sumit/bharatrag/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")

def setup_retriever():
    print("Loading ChromaDB...")
    embeddings  = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"Loaded {vectorstore._collection.count()} chunks")
    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

retriever = setup_retriever()

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    anthropic_api_key=ANTHROPIC_KEY
)

# ============================================
# STATE — New field: rewritten_question
# ============================================
# Compare to Day 6 state:
# Day 6: question, documents, answer, grade, attempts
# Day 7: adds rewritten_question
#
# rewritten_question starts empty
# query_rewrite_node fills it on first failure
# retrieve_node uses it instead of original question
# on second attempt
# ============================================

class RAGState(TypedDict):
    question:           str
    rewritten_question: str   # NEW — Claude's rewrite
    documents:          List[str]
    answer:             str
    grade:              str
    attempts:           int

# ============================================
# NODES
# ============================================

def retrieve_node(state: RAGState) -> dict:
    """
    Node 1: Retrieve from ChromaDB.

    SMART PART:
    First attempt  → uses original question
    Second attempt → uses rewritten_question
    This is why rewriting helps — different
    query hits different vectors in ChromaDB
    """
    attempts = state.get("attempts", 0)

    # Use rewritten question on retry if available
    rewritten = state.get("rewritten_question", "")
    if attempts > 0 and rewritten:
        query = rewritten
        print(f"  [retrieve] Using rewritten query: '{query}'")
    else:
        query = state["question"]
        print(f"  [retrieve] Using original query: '{query}'")

    docs      = retriever.invoke(query)
    doc_texts = [doc.page_content for doc in docs]

    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        print(f"    Chunk {i+1} "
              f"(Page {int(page)+1}): "
              f"{doc.page_content[:60]}...")

    return {
        "documents": doc_texts,
        "attempts":  attempts + 1
    }


def grade_node(state: RAGState) -> dict:
    """
    Node 2: Grade relevance.(relevance grading)
    Same as Day 6 — checks if chunks answer question.
    """
    question  = state["question"]
    documents = state["documents"]
    print(f"  [grade] Checking relevance...")

    grade_prompt = f"""Are these documents relevant to answer: "{question}"?

Documents:
{chr(10).join(documents)}

Reply ONLY with: relevant or irrelevant"""
    #sends the grading task to the LLM and expects a simple relevant/irrelevant response
    response = llm.invoke(
        [HumanMessage(content=grade_prompt)]
    )
    grade  = response.content.strip().lower()
    result = "irrelevant" if "irrelevant" in grade else "relevant"

    print(f"  [grade] Result: {result}")
    return {"grade": result}


def query_rewrite_node(state: RAGState) -> dict:
    """
    Node 3 (NEW): Rewrite the query when retrieval fails.

    This is the REFLEXION step.
    Instead of retrying with same query:
    1. Ask Claude WHY the query might have failed
    2. Ask Claude to rephrase it differently
    3. Store rewritten query in state
    4. retrieve_node will use this on next attempt

    The key insight:
    "What is session state?"
    might not match chunks that say
    "st.session_state persists across reruns"

    Rewrite: "Streamlit session state st.session_state persist"
    Now hits the right vectors.
    """
    original  = state["question"]
    documents = state["documents"]

    print(f"  [rewrite] Original query failed. Rewriting...")

    rewrite_prompt = f"""The following question failed to retrieve 
relevant documents from a technical document about 
Python, Streamlit, and AI development.

Original question: "{original}"

Retrieved (but irrelevant) documents:
{chr(10).join(documents[:2])}

Task: Rewrite the question to be more specific and 
technical so it finds better matching content.

Rules:
- Use technical keywords likely to appear in the document
- Keep it concise — under 15 words
- Focus on the core concept being asked about
- Do NOT add explanation — just the rewritten question

Rewritten question:"""

    response  = llm.invoke(
        [HumanMessage(content=rewrite_prompt)]
    )
    rewritten = response.content.strip()

    # Clean up — remove quotes if Claude added them
    rewritten = rewritten.strip('"').strip("'")

    print(f"  [rewrite] New query: '{rewritten}'")
    return {"rewritten_question": rewritten}


def generate_node(state: RAGState) -> dict:
    """
    Node 4: Generate answer from relevant chunks.
    Same as Day 6.
    """
    question  = state["question"]
    documents = state["documents"]
    print(f"  [generate] Creating answer...")

    context = "\n\n".join(documents)
    #Grounded Generation: instructs the LLM to answer using only the retrieved documents as 
    # context, and to be concise and cite page numbers when possible. If the answer 
    # cannot be found in the documents, it should explicitly say so.
    prompt  = f"""Answer using ONLY this context.
If not found say: "I could not find this in the document."
Be concise and cite page numbers when possible.

Context: {context}

Question: {question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}

#Fallback / Guardrail / Safe Failure Node:
def fallback_node(state: RAGState) -> dict:
    """
    Node 5: Honest fallback after all attempts fail.
    Now shows both original and rewritten queries
    so user understands what was tried.
    """
    original  = state["question"]
    rewritten = state.get("rewritten_question", "")
    attempts  = state.get("attempts", 0)

    print(f"  [fallback] All {attempts} attempts failed.")

    if rewritten:
        answer = (
            f"I searched for '{original}' and also tried "
            f"'{rewritten}' but could not find relevant "
            f"information in the document. "
            f"Please ask about topics covered in the document."
        )
    else:
        answer = (
            "I could not find relevant information "
            "to answer your question. Please try "
            "rephrasing or ask about a different topic."
        )

    return {"answer": answer}


# ============================================
# ROUTER creates CONTROL FLOW
# ============================================
# Day 6 router:
# relevant  → generate
# irrelevant + attempts >= 2 → fallback
# irrelevant + attempts < 2  → retrieve (same query)
#
# Day 7 router:
# relevant  → generate
# irrelevant + attempts == 1 → rewrite (NEW step)
# irrelevant + attempts >= 2 → fallback
#
# The difference: instead of blindly retrying,
# we first rewrite, then retry with new query
# ============================================

def should_generate(state: RAGState) -> str:
    """
    Router: decides next step after grading.

    Flow:
    Attempt 1 fails → rewrite query
    Attempt 2 fails → fallback
    Any attempt succeeds → generate
    """
    grade    = state.get("grade", "irrelevant")
    attempts = int(state.get("attempts", 0))

    print(f"  [router] grade={grade} attempts={attempts}")

    if grade == "relevant":
        print("  [router] Relevant → generate")
        return "generate"

    elif attempts == 1:
        print("  [router] Failed attempt 1 → rewriting query")
        return "rewrite"

    else:
        print("  [router] All attempts failed → fallback")
        return "fallback"


# ============================================
# BUILD GRAPH
# ============================================
# Day 6 graph:
# retrieve → grade → [router] → generate/fallback/retrieve
#
# Day 7 graph:
# retrieve → grade → [router] → generate
#                             → rewrite → retrieve → grade
#                             → fallback
#
# rewrite_node is inserted between grade and retrieve
# Creates: grade → rewrite → retrieve → grade loop
# ============================================

def build_agent():
    workflow = StateGraph(RAGState)

    # Register all nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade",    grade_node)
    workflow.add_node("rewrite",  query_rewrite_node)  # NEW
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)

    # Fixed edges
    workflow.add_edge("retrieve", "grade")
    # After retrieve → always grade

    workflow.add_edge("rewrite", "retrieve")
    # After rewrite → retrieve with new query
    # This creates the smart retry loop

    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # Conditional edge from grade
    workflow.add_conditional_edges(
        "grade",
        should_generate,
        {
            "generate": "generate",
            "rewrite":  "rewrite",   # NEW
            "fallback": "fallback",
        }
    )

    workflow.set_entry_point("retrieve")
    return workflow.compile()


# ============================================
# RUN
# ============================================

def run_agent(question: str, agent) -> str:
    print(f"\nQuestion: '{question}'")
    print("=" * 55)

    initial_state = {
        "question":           question,
        "rewritten_question": "",      # empty at start
        "documents":          [],
        "answer":             "",
        "grade":              "irrelevant",
        "attempts":           0,
    }

    final_state = agent.invoke(initial_state)

    print(f"\nAnswer: {final_state['answer']}")
    print(f"Attempts: {final_state['attempts']}")

    if final_state.get("rewritten_question"):
        print(f"Rewritten query: "
              f"'{final_state['rewritten_question']}'")

    return final_state["answer"]


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("Building Day 7 LangGraph agent...")
    agent = build_agent()
    print("Agent ready!")
    print()

    questions = [
        # Should work first try — in document
        "What is session state in Streamlit?",

        # Should work first try — in document
        "What is the difference between V1 and V2?",

        # Might need rewrite — vague question
        "How does the app remember things?",

        # Should need rewrite — not obvious keywords
        "What happens when too many people use it?",

        # Should fallback — truly not in document
        "How to make biryani?",
    ]

    for q in questions:
        run_agent(q, agent)
        print()




"""
Building Day 7 LangGraph agent...
Agent ready!


Question: 'What is session state in Streamlit?'
=======================================================
  [retrieve] Using original query: 'What is session state in Streamlit?'
    Chunk 1 (Page 2): send the entire conversation history every time — to simulat...
    Chunk 2 (Page 2): means normal variables reset every rerun. Only st.session_st...
    Chunk 3 (Page 9): doesn't already exist in session state. Adding a new session...
  [grade] Checking relevance...
  [grade] Result: relevant
  [router] grade=relevant attempts=1
  [router] Relevant → generate
  [generate] Creating answer...

Answer: **Session state in Streamlit** is a dictionary (`st.session_state`) that persists across reruns. Anything stored in it survives the next rerun, allowing the application to remember data like uploaded PDFs and chat history between user interactions.

This is necessary because every user action causes Streamlit to rerun the entire Python file from line 1, which means normal variables reset every rerun. Only `st.session_state` survives across these reruns.
Attempts: 1


Question: 'What is the difference between V1 and V2?'
=======================================================
  [retrieve] Using original query: 'What is the difference between V1 and V2?'
    Chunk 1 (Page 8): VERSION 2 — PRODUCTION READY
V2 — Error Handling, Security, ...
    Chunk 2 (Page 1): V1 — Prototype
Core working app. 50 lines. Upload PDF, ask C...
    Chunk 3 (Page 3): VERSION 1 — THE PROTOTYPE
V1 — Core Working App
V1 is the mi...
  [grade] Checking relevance...
  [grade] Result: relevant
  [router] grade=relevant attempts=1
  [router] Relevant → generate
  [generate] Creating answer...

Answer: Based on the document, the key differences between V1 and V2 are:

**V1 (Prototype):**
- Core working app with ~50 lines of code
- Basic functionality: upload PDF, ask question, get answer
- No error handling, no security, no polish
- Not for real users

**V2 (Production Ready):**
- Adds three major features:
  1. **Error handling** - shows friendly messages instead of crashing
  2. **Client API key input** - clients enter their own Anthropic key so you pay nothing
  3. **File validation** - catches bad uploads gracefully
- Includes UI polish
- Ready for real users

In summary, V2 transforms the V1 prototype into a production-ready application by adding proper error handling, client API key management, and security features.
Attempts: 1


Question: 'How does the app remember things?'
=======================================================
  [retrieve] Using original query: 'How does the app remember things?'
    Chunk 1 (Page 6): document answers.
messages=chat_history — sends the full con...
    Chunk 2 (Page 2): top to bottom.
Claude API
Anthropic's AI model accessed prog...
    Chunk 3 (Page 2): means normal variables reset every rerun. Only st.session_st...
  [grade] Checking relevance...
  [grade] Result: relevant
  [router] grade=relevant attempts=1
  [router] Relevant → generate
  [generate] Creating answer...

Answer: The app remembers things through **st.session_state**, which is a dictionary that persists across reruns. Anything stored here survives the next rerun, allowing SmartDocs to remember uploaded PDFs and chat history between questions.

Additionally, since the Claude API is **stateless** (remembers nothing between calls), the app simulates memory by manually sending the **entire conversation history** (messages=chat_history) with every API call.

*Page numbers not provided in context.*
Attempts: 1


Question: 'What happens when too many people use it?'
=======================================================
  [retrieve] Using original query: 'What happens when too many people use it?'
    Chunk 1 (Page 8): VERSION 2 — PRODUCTION READY
V2 — Error Handling, Security, ...
    Chunk 2 (Page 11): user.
st.stop() — stops all code execution immediately. Noth...
    Chunk 3 (Page 8): New: API key input
Clients enter their own Anthropic key — y...
  [grade] Checking relevance...
  [grade] Result: irrelevant
  [router] grade=irrelevant attempts=1
  [router] Failed attempt 1 → rewriting query
  [rewrite] Original query failed. Rewriting...
  [rewrite] New query: 'How does the application handle API rate limits when usage is high?'
  [retrieve] Using rewritten query: 'How does the application handle API rate limits when usage is high?'
    Chunk 1 (Page 12): except anthropic.APIStatusError as e:
st.error(f'API error (...
    Chunk 2 (Page 8): VERSION 2 — PRODUCTION READY
V2 — Error Handling, Security, ...
    Chunk 3 (Page 11): user.
st.stop() — stops all code execution immediately. Noth...
  [grade] Checking relevance...
  [grade] Result: relevant
  [router] grade=relevant attempts=2
  [router] Relevant → generate
  [generate] Creating answer...

Answer: When too many requests are made too fast, a `RateLimitError` occurs. The app tells the user to wait and displays the message: "Rate limit reached. Wait a moment and try again."
Attempts: 2
Rewritten query: 'How does the application handle API rate limits when usage is high?'


Question: 'How to make biryani?'
=======================================================
  [retrieve] Using original query: 'How to make biryani?'
    Chunk 1 (Page 24): The Development Philosophy
V1 → Ship something working
V2 → ...
    Chunk 2 (Page 4): ■■ NEVER put this file on GitHub. Add .env to your .gitignor...
    Chunk 3 (Page 1): SmartDocs AI
 Complete Developer Learning Guide
 V1 → V2 → V...
  [grade] Checking relevance...
  [grade] Result: irrelevant
  [router] grade=irrelevant attempts=1
  [router] Failed attempt 1 → rewriting query
  [rewrite] Original query failed. Rewriting...
  [rewrite] New query: 'What is the development philosophy and version progression for building production applications?'
  [retrieve] Using rewritten query: 'What is the development philosophy and version progression for building production applications?'
    Chunk 1 (Page 24): The Development Philosophy
V1 → Ship something working
V2 → ...
    Chunk 2 (Page 1): V1 — Prototype
Core working app. 50 lines. Upload PDF, ask C...
    Chunk 3 (Page 1): SmartDocs AI
 Complete Developer Learning Guide
 V1 → V2 → V...
  [grade] Checking relevance...
  [grade] Result: irrelevant
  [router] grade=irrelevant attempts=2
  [router] All attempts failed → fallback
  [fallback] All 2 attempts failed.

Answer: I searched for 'How to make biryani?' and also tried 'What is the development philosophy and version progression for building production applications?' but could not find relevant information in the document. Please ask about topics covered in the document.
Attempts: 2
Rewritten query: 'What is the development philosophy and version progression for building production applications?'

"""        