# ============================================
# DAY 8: Hallucination Detection
# LLM-as-judge pattern
# ============================================
# DAY 7 RECAP:
# retrieve → grade → rewrite → retrieve → generate
# Problem: no check if answer is actually correct
#
# DAY 8 ADDS:
# ... → generate → hallucination_check
#                → grounded    → return answer
#                → hallucinated → regenerate
#
# LLM-as-judge:
# Claude evaluates Claude's own answer
# Different prompt = different perspective
# Catches unsupported claims before user sees them
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

# ── Setup ─────────────────────────────────────────────
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
# STATE — Two new fields
# ============================================

class RAGState(TypedDict):
    question:            str
    rewritten_question:  str
    documents:           List[str]
    answer:              str
    grade:               str        # relevant/irrelevant
    hallucination:       str        # NEW: grounded/hallucinated
    attempts:            int        # retrieval attempts
    generation_attempts: int        # NEW: generation attempts

# ============================================
# NODES — From Day 7 (unchanged)
# ============================================

def retrieve_node(state: RAGState) -> dict:
    """Retrieve from ChromaDB — uses rewrite if available."""
    attempts = state.get("attempts", 0)
    rewritten = state.get("rewritten_question", "")

    if attempts > 0 and rewritten:
        query = rewritten
        print(f"  [retrieve] Rewritten query: '{query}'")
    else:
        query = state["question"]
        print(f"  [retrieve] Original query: '{query}'")

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
    """Grade relevance of retrieved chunks."""
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
    """Rewrite query when retrieval fails — Day 7."""
    original  = state["question"]
    documents = state["documents"]
    print(f"  [rewrite] Rewriting failed query...")

    prompt = f"""Question failed to retrieve relevant docs: "{original}"

Retrieved (irrelevant) docs:
{chr(10).join(documents[:2])}

Rewrite as a more specific technical question.
Under 15 words. Just the question, no explanation.

Rewritten question:"""

    response  = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip().strip('"').strip("'")
    print(f"  [rewrite] New query: '{rewritten}'")
    return {"rewritten_question": rewritten}


def generate_node(state: RAGState) -> dict:
    """
    Generate answer from retrieved chunks.

    Day 8 change: tracks generation_attempts
    So hallucination checker knows if this
    is first attempt or a regeneration attempt.

    On regeneration (gen_attempts > 0):
    Uses STRICTER prompt that emphasizes
    staying within the context only.
    """
    question         = state["question"]
    documents        = state["documents"]
    gen_attempts     = state.get("generation_attempts", 0)

    print(f"  [generate] Creating answer "
          f"(generation attempt {gen_attempts + 1})...")

    context = "\n\n".join(documents)

    if gen_attempts == 0:
        # First attempt — normal prompt
        prompt = f"""Answer using ONLY this context.
If not found: "I could not find this in the document."
Be concise and cite page numbers.

Context: {context}

Question: {question}

Answer:"""
    else:
        # Regeneration — stricter prompt
        # Previous answer was hallucinated
        # Force Claude to stay strictly in context
        prompt = f"""STRICT INSTRUCTIONS:
Your previous answer contained information NOT in the context.
Answer using ONLY exact information from the context below.
Do NOT add any outside knowledge.
If the exact answer is not in context, say:
"I could not find this specific information in the document."

Context: {context}

Question: {question}

Strictly grounded answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "answer":             response.content,
        "generation_attempts": gen_attempts + 1
    }

#Hallucination check node is the key addition for Day 8. 
# It implements the LLM-as-judge pattern, where we ask Claude to evaluate its own answer against the retrieved context. 
# This allows us to catch unsupported claims before they reach the user, improving the reliability of our agent.
def hallucination_check_node(state: RAGState) -> dict:
    """
    Node (NEW): Check if answer is grounded in context.

    LLM-as-judge pattern:
    Send context + answer to Claude
    Ask Claude to verify the answer

    This is the key Day 8 addition.
    A chain NEVER checks its own output.
    An agent VERIFIES before returning.

    READS:   state["answer"], state["documents"]
    UPDATES: state["hallucination"]
    NEXT:    conditional — grounded or hallucinated
    """
    answer    = state["answer"]
    documents = state["documents"]
    question  = state["question"]

    print(f"  [hallucination_check] Verifying answer...")

    check_prompt = f"""You are a fact-checker.

Check if this answer is fully supported by the context.

Context:
{chr(10).join(documents)}

Question: {question}

Answer to verify: {answer}

Is every claim in the answer directly supported
by the context above?

Reply ONLY with: grounded or hallucinated

- grounded    = answer only contains information from context
- hallucinated = answer contains information NOT in context"""

    response = llm.invoke(
        [HumanMessage(content=check_prompt)]
    )
    result = response.content.strip().lower()

    if "hallucinated" in result:
        hallucination = "hallucinated"
    else:
        hallucination = "grounded"

    print(f"  [hallucination_check] Result: {hallucination}")
    return {"hallucination": hallucination}


def fallback_node(state: RAGState) -> dict:
    """Honest fallback — shows what was tried."""
    original  = state["question"]
    rewritten = state.get("rewritten_question", "")
    attempts  = state.get("attempts", 0)

    print(f"  [fallback] All attempts exhausted.")

    if rewritten:
        answer = (
            f"I searched for '{original}' and also "
            f"'{rewritten}' but could not find "
            f"relevant information in the document."
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
    """
    Router after grade_node.
    Same as Day 7 — adds rewrite step.
    """
    grade    = state.get("grade", "irrelevant")
    attempts = int(state.get("attempts", 0))

    print(f"  [router] grade={grade} attempts={attempts}")

    if grade == "relevant":
        print("  [router] Relevant → generate")
        return "generate"
    elif attempts == 1:
        print("  [router] Failed → rewriting query")
        return "rewrite"
    else:
        print("  [router] All retrieval failed → fallback")
        return "fallback"


def should_return(state: RAGState) -> str:
    """
    Router (NEW): after hallucination_check_node.
    Decides: return answer OR regenerate?

    grounded     → return the answer (END)
    hallucinated → regenerate with stricter prompt
                   but only if generation_attempts < 2
                   prevents infinite regeneration loop
    """
    hallucination    = state.get("hallucination", "grounded")
    gen_attempts     = int(state.get(
        "generation_attempts", 0
    ))

    print(f"  [router] hallucination={hallucination} "
          f"gen_attempts={gen_attempts}")

    if hallucination == "grounded":
        print("  [router] Grounded → return answer")
        return "return_answer"

    elif gen_attempts < 2:
        print("  [router] Hallucinated → regenerate")
        return "regenerate"

    else:
        print("  [router] Max regenerations → return anyway")
        return "return_answer"


# ============================================
# BUILD GRAPH
# ============================================
# Day 7 graph:
# retrieve → grade → rewrite → retrieve (loop)
#                 → generate → END
#                 → fallback → END
#
# Day 8 graph:
# retrieve → grade → rewrite → retrieve (loop)
#                 → generate → hallucination_check
#                                → grounded    → END
#                                → hallucinated → generate (retry)
#                 → fallback → END
# ============================================

def build_agent():
    workflow = StateGraph(RAGState)

    # Register all nodes
    workflow.add_node("retrieve",            retrieve_node)
    workflow.add_node("grade",               grade_node)
    workflow.add_node("rewrite",             query_rewrite_node)
    workflow.add_node("generate",            generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("fallback",            fallback_node)

    # Fixed edges
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("rewrite",  "retrieve")
    workflow.add_edge("generate", "hallucination_check")
    # After generate → always check for hallucination
    # This is new — Day 7 went straight to END

    workflow.add_edge("fallback", END)

    # Conditional edge after grade
    workflow.add_conditional_edges(
        "grade",
        should_generate,
        {
            "generate": "generate",
            "rewrite":  "rewrite",
            "fallback": "fallback",
        }
    )

    # Conditional edge after hallucination check (NEW)
    workflow.add_conditional_edges(
        "hallucination_check",
        should_return,
        {
            "return_answer": END,
            "regenerate":    "generate",
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
        "question":            question,
        "rewritten_question":  "",
        "documents":           [],
        "answer":              "",
        "grade":               "irrelevant",
        "hallucination":       "grounded",
        "attempts":            0,
        "generation_attempts": 0,
    }

    final_state = agent.invoke(initial_state)

    print(f"\nFinal Answer: {final_state['answer']}")
    print(f"Retrieval attempts:  {final_state['attempts']}")
    print(f"Generation attempts: "
          f"{final_state['generation_attempts']}")
    print(f"Hallucination:       "
          f"{final_state.get('hallucination', 'N/A')}")

    return final_state["answer"]


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("Building Day 8 LangGraph agent...")
    agent = build_agent()
    print("Agent ready!")
    print()

    questions = [
        # Should pass hallucination check easily
        "What is session state in Streamlit?",

        # Should pass — factual and in document
        "What is the difference between V1 and V2?",

        # Interesting — vague, needs rewrite
        "How does the app handle errors?",

        # Should fallback — not in document
        "How to make biryani?",
    ]

    for q in questions:
        run_agent(q, agent)
        print()

"""###
Industry Relevance

This exact pattern is now used in:

AI copilots
legal AI
medical AI
enterprise assistants
coding agents

because hallucination control is critical.
###"""


"""

Building Day 8 LangGraph agent...
Agent ready!


Question: 'What is session state in Streamlit?'
=======================================================
  [retrieve] Original query: 'What is session state in Streamlit?'
    Chunk 1 (Page 2): send the entire conversation history every time — to simulat...
    Chunk 2 (Page 2): means normal variables reset every rerun. Only st.session_st...
    Chunk 3 (Page 9): doesn't already exist in session state. Adding a new session...
  [grade] Checking relevance...
  [grade] Result: relevant
  [router] grade=relevant attempts=1
  [router] Relevant → generate
  [generate] Creating answer (generation attempt 1)...
  [hallucination_check] Verifying answer...
  [hallucination_check] Result: grounded
  [router] hallucination=grounded gen_attempts=1
  [router] Grounded → return answer

Final Answer: **Session state in Streamlit** is a dictionary (`st.session_state`) that persists across reruns. Anything stored in it survives the next rerun, allowing the application to remember data like uploaded PDFs and chat history between user interactions.

Normal variables reset every rerun, but `st.session_state` is the mechanism that provides memory across reruns.

*Page numbers not provided in context.*
Retrieval attempts:  1
Generation attempts: 1
Hallucination:       grounded


Question: 'What is the difference between V1 and V2?'
=======================================================
  [retrieve] Original query: 'What is the difference between V1 and V2?'
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
  [generate] Creating answer (generation attempt 1)...
  [hallucination_check] Verifying answer...
  [hallucination_check] Result: grounded
  [router] hallucination=grounded gen_attempts=1
  [router] Grounded → return answer

Final Answer: Based on the context provided:

**V1 (Prototype):**
- Core working app with ~50 lines of code
- Basic functionality: upload PDF, ask question, get answer
- No error handling, no security, no polish
- Not for real users

**V2 (Production Ready):**
- Adds proper error handling (friendly messages instead of crashes)
- Adds client API key input (clients use their own Anthropic key)
- Adds file validation for uploads
- Includes UI polish
- Ready for real users

The main difference is that V2 transforms the prototype into a production-ready application by adding error handling, security features, and the ability for clients to use their own API keys.

(Page numbers not provided in the context)
Retrieval attempts:  1
Generation attempts: 1
Hallucination:       grounded


Question: 'How does the app handle errors?'
=======================================================
  [retrieve] Original query: 'How does the app handle errors?'
    Chunk 1 (Page 8): VERSION 2 — PRODUCTION READY
V2 — Error Handling, Security, ...
    Chunk 2 (Page 17): 100/100
FINAL SCORE
100/100 — Grade A: Production Ready
Test...
    Chunk 3 (Page 1): V1 — Prototype
Core working app. 50 lines. Upload PDF, ask C...
  [grade] Checking relevance...
  [grade] Result: relevant
  [router] grade=relevant attempts=1
  [router] Relevant → generate
  [generate] Creating answer (generation attempt 1)...
  [hallucination_check] Verifying answer...
  [hallucination_check] Result: grounded
  [router] hallucination=grounded gen_attempts=1
  [router] Grounded → return answer

Final Answer: According to the document, in **Version 2**, the app implements proper error handling where "Every API error shows a friendly message instead of crashing" rather than showing a red crash screen (page not explicitly numbered, but found in the V2 section).
Retrieval attempts:  1
Generation attempts: 1
Hallucination:       grounded


Question: 'How to make biryani?'
=======================================================
  [retrieve] Original query: 'How to make biryani?'
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
  [router] Failed → rewriting query
  [rewrite] Rewriting failed query...
  [rewrite] New query: 'What causes semantic search to return irrelevant documents for unrelated queries?'
  [retrieve] Rewritten query: 'What causes semantic search to return irrelevant documents for unrelated queries?'
    Chunk 1 (Page 6): document answers.
messages=chat_history — sends the full con...
    Chunk 2 (Page 6): what it can/cannot do. We inject the entire PDF text here. T...
    Chunk 3 (Page 17): 100/100
FINAL SCORE
100/100 — Grade A: Production Ready
Test...
  [grade] Checking relevance...
  [grade] Result: irrelevant
  [router] grade=irrelevant attempts=2
  [router] All retrieval failed → fallback
  [fallback] All attempts exhausted.

Final Answer: I searched for 'How to make biryani?' and also 'What causes semantic search to return irrelevant documents for unrelated queries?' but could not find relevant information in the document.
Retrieval attempts:  2
Generation attempts: 0
Hallucination:       grounded


"""