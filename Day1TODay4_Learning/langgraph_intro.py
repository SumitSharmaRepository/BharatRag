# ============================================
# DAY 6: LangGraph Agentic RAG
# ============================================
# WHAT IS LANGGRAPH?
# LangGraph builds stateful agents as graphs.
# A graph has:
#   NODES  = functions that do work
#   EDGES  = connections between nodes
#   STATE  = shared memory across all nodes
#
# DIFFERENCE FROM LANGCHAIN CHAIN (Day 3-5):
#   Chain:  fixed path A→B→C→D always
#   Agent:  dynamic path based on decisions
#           A→B→C or A→B→A→B→D depending on state
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()
# load_dotenv() reads .env file into os.environ
# Same as always — API keys loaded first

from typing import TypedDict, List
# TypedDict: creates a dictionary with type hints
# Like a Java class but lighter weight
# Tells Python AND other developers what keys exist
# List: type hint for Python lists

from langchain_anthropic import ChatAnthropic
# Claude API wrapper from LangChain
# Same as Days 3-5

from langgraph.graph import StateGraph, END
# StateGraph: the main graph builder class
#   You add nodes and edges to it
#   Then compile() it into a runnable agent
#
# END: special constant that means "stop here"
#   When a node connects to END → graph finishes
#   Like return in a function but for the whole graph

from langchain_chroma import Chroma
# ChromaDB vector store — updated import
# langchain-chroma is the new package (not community)
# Same functionality as Day 2-5

from langchain_huggingface import HuggingFaceEmbeddings
# Embedding model wrapper
# Converts text to vectors (Day 1 concept)
# Used to embed queries before searching ChromaDB

from langchain_core.messages import HumanMessage
# Wraps a string as a "user" message for Claude
# Claude expects messages in this format:
# [HumanMessage("your text")] not just "your text"

# ============================================
# CHROMADB SETUP
# ============================================
CHROMA_PATH     = "/home/sumit/bharatrag/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")
# Constants at top = easy to change in one place
# Java equivalent: static final String CHROMA_PATH = "..."

def setup_retriever():
    """
    Load existing ChromaDB and return a retriever.
    Called ONCE at startup — not on every question.

    Returns:
        retriever object that searches ChromaDB
    """
    print("Loading ChromaDB...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    # Creates embedding model
    # Same model used when PDFs were indexed (Day 4-5)
    # MUST be identical — different model = wrong vectors
    # Java equivalent: instantiating a service class

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    # Loads existing ChromaDB from disk
    # Does NOT re-index — just opens the database
    # persist_directory tells it WHERE to find the DB

    print(f"Loaded {vectorstore._collection.count()} chunks")
    # _collection.count() = how many chunks stored
    # Should print 71 (from Day 4-5)

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    # as_retriever() wraps the vectorstore
    # so it can be used in LangChain/LangGraph
    # k=3 means: return TOP 3 most similar chunks
    # per semantic search query

retriever = setup_retriever()
# Called once at module level
# retriever is shared across ALL nodes
# No need to reload ChromaDB on every question

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    anthropic_api_key=ANTHROPIC_KEY
)
# Claude LLM instance — also shared
# temperature=0 = deterministic answers
# Same response every time for same input
# Good for factual Q&A, bad for creative writing

# ============================================
# STATE DEFINITION
# ============================================
# STATE is the most important concept in LangGraph
#
# State = shared memory that flows through the graph
# Every node READS from state
# Every node WRITES back to state
# State persists across all nodes in one run
#
# TypedDict ensures every key has a known type
# Like defining fields in a Java class
# ============================================

class RAGState(TypedDict):
    question:  str        # user's original question
                          # set at start, never changes

    documents: List[str]  # chunks retrieved from ChromaDB
                          # set by retrieve_node
                          # read by grade_node and generate_node

    answer:    str        # final answer to return to user
                          # set by generate_node or fallback_node
                          # empty string until one of these runs

    grade:     str        # "relevant" or "irrelevant"
                          # set by grade_node
                          # read by should_generate router
                          # using string not bool (LangGraph bug workaround)

    attempts:  int        # how many times retrieve_node has run
                          # incremented by retrieve_node each call
                          # router uses this to trigger fallback
                          # prevents infinite retry loops

# ============================================
# NODES
# ============================================
# A NODE is a Python function that:
#   1. Receives FULL current state as input
#   2. Does some work (retrieval, grading, etc)
#   3. Returns PARTIAL state update (dict)
#      Only return keys you changed
#      Other keys stay unchanged automatically
#
# Java equivalent: a step in a workflow/pipeline
# Each step reads shared context, updates it
# ============================================

def retrieve_node(state: RAGState) -> dict:
    """
    Node 1: Search ChromaDB for relevant chunks.

    READS:   state["question"], state["attempts"]
    UPDATES: state["documents"], state["attempts"]
    NEXT:    always goes to grade_node (fixed edge)
    """
    question = state["question"]
    print(f"  [retrieve] Searching: '{question}'")

    docs = retriever.invoke(question)
    # retriever.invoke() does three things internally:
    # 1. Embeds question → [0.2, 0.8, 0.1, ...]
    # 2. Searches ChromaDB for similar vectors
    # 3. Returns top k=3 Document objects

    doc_texts = [doc.page_content for doc in docs]
    # Extract just the text from Document objects
    # doc.page_content = the actual text chunk
    # doc.metadata = {"source": "file.pdf", "page": 3}
    # List comprehension = [expression for item in list]
    # Java equivalent: stream().map().collect()

    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        print(f"    Chunk {i+1} (Page {int(page)+1}): "
              f"{doc.page_content[:60]}...")
    # enumerate() gives index AND value
    # page+1 because PDF pages are 0-indexed internally
    # [:60] shows only first 60 chars for readability

    return {
        "documents": doc_texts,
        "attempts":  state.get("attempts", 0) + 1
    }
    # Return ONLY what changed — LangGraph merges this
    # with existing state automatically
    # attempts + 1 tracks how many times we retrieved
    # This prevents infinite loops in the router


def grade_node(state: RAGState) -> dict:
    """
    Node 2: Ask Claude if retrieved chunks are relevant.

    This is the KEY difference from a chain.
    Chain: blindly sends chunks to generate
    Agent: CHECKS if chunks are actually useful

    READS:   state["question"], state["documents"]
    UPDATES: state["grade"]
    NEXT:    conditional — depends on grade result
             goes to should_generate() router
    """
    question  = state["question"]
    documents = state["documents"]
    print(f"  [grade] Checking relevance...")

    grade_prompt = f"""Are these documents relevant to answer: "{question}"?

Documents:
{chr(10).join(documents)}

Reply ONLY with the word: relevant or irrelevant"""
    # chr(10) = newline character (\n)
    # join() combines list items with that separator
    # Strict prompt: "ONLY with the word" reduces hallucination

    response = llm.invoke([HumanMessage(content=grade_prompt)])
    # [HumanMessage(...)] = list with one message
    # LangChain expects a list of messages
    # HumanMessage marks it as coming from the user

    grade = response.content.strip().lower()
    # .strip() removes whitespace/newlines from edges
    # .lower() makes comparison case-insensitive
    # "Relevant" == "relevant" == "RELEVANT" all work

    if "irrelevant" in grade:
        result = "irrelevant"
    else:
        result = "relevant"
    # Check for "irrelevant" FIRST
    # Because "irrelevant" contains "relevant"
    # If you checked "relevant" first:
    # "irrelevant" would match "relevant" → BUG

    print(f"  [grade] Result: {result}")
    return {"grade": result}
    # Using string "relevant"/"irrelevant" not bool
    # LangGraph bug: False bool sometimes ignored in state
    # String never gets ignored → reliable routing


def generate_node(state: RAGState) -> dict:
    """
    Node 3: Generate answer using Claude.
    Only runs when grade_node marked chunks as relevant.

    READS:   state["question"], state["documents"]
    UPDATES: state["answer"]
    NEXT:    always END (fixed edge)
    """
    question  = state["question"]
    documents = state["documents"]
    print(f"  [generate] Creating answer...")

    context = "\n\n".join(documents)
    # Join all chunks with double newline
    # Double newline = clear visual separation
    # Claude understands this as separate sections

    prompt = f"""Answer using ONLY this context.
If not found say: "I could not find this in the document."

Context: {context}

Question: {question}

Answer:"""
    # f-string embeds context and question
    # "ONLY this context" prevents hallucination
    # Claude stays within retrieved information

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}
    # response.content = Claude's text response
    # Stored in state["answer"] for final output


def fallback_node(state: RAGState) -> dict:
    """
    Node 4: Honest answer when retrieval fails.

    Runs when:
    - Chunks graded irrelevant AND attempts >= 2
    - The agent tried, failed, admits it

    A chain has NO fallback.
    Agent handles failure gracefully.

    READS:   nothing from state
    UPDATES: state["answer"]
    NEXT:    always END (fixed edge)
    """
    print(f"  [fallback] No relevant chunks found.")
    return {
        "answer": (
            "I could not find relevant information "
            "to answer your question. Please try "
            "rephrasing or ask about a topic in the document."
        )
    }

# ============================================
# ROUTER
# ============================================
# Router = function that returns a STRING
# That string tells LangGraph which node to go to next
#
# This is conditional_edges in action:
# Instead of fixed A→B, you have A→?(depends on state)
#
# The router reads state and decides:
# "relevant"  → go to generate node
# "irrelevant + attempts < 2" → go back to retrieve
# "irrelevant + attempts >= 2" → go to fallback
# ============================================

def should_generate(state: RAGState) -> str:
    """
    Router: called after grade_node.
    Reads grade and attempts from state.
    Returns name of next node to execute.

    Return values MUST match keys in
    add_conditional_edges() mapping dict.
    """
    grade    = state.get("grade", "irrelevant")
    attempts = int(state.get("attempts", 0))
    # .get("key", default) = safe dictionary access
    # If key missing, returns default instead of error
    # int() ensures attempts is always a number

    print(f"  [router] grade={grade} attempts={attempts}")

    if grade == "relevant":
        print("  [router] Relevant → generate")
        return "generate"
        # Chunks are good → generate answer

    elif attempts >= 2:
        print("  [router] Max attempts → fallback")
        return "fallback"
        # Tried twice, still irrelevant → give up honestly

    else:
        print("  [router] Irrelevant → retry")
        return "retrieve"
        # First attempt failed → try retrieving again
        # retrieve_node will run again with same question
        # Day 7 will add query rewriting here

# ============================================
# BUILD THE GRAPH
# ============================================
# StateGraph is built in 5 steps:
# 1. Create graph with state schema
# 2. Add nodes (register functions)
# 3. Add fixed edges (A always goes to B)
# 4. Add conditional edges (A goes to ? based on router)
# 5. Set entry point (where execution starts)
# 6. Compile (build executable agent)
# ============================================

def build_agent():
    """
    Assemble and compile the LangGraph agent.

    Graph structure:
    START
      ↓
    retrieve → grade → [should_generate router]
                              ↓           ↓         ↓
                          generate    retrieve   fallback
                              ↓                      ↓
                             END                    END
    """

    workflow = StateGraph(RAGState)
    # StateGraph takes your TypedDict as schema
    # It knows what keys state should have
    # Validates state updates against this schema

    # Register nodes — give each function a name
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade",    grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)
    # First arg:  string name used in edges
    # Second arg: the actual Python function
    # Name and function can differ but keep them similar

    # Fixed edges — these ALWAYS happen
    workflow.add_edge("retrieve", "grade")
    # After retrieve_node runs → always go to grade_node
    # No conditions, no choices

    workflow.add_edge("generate", END)
    # After generate_node runs → always stop
    # END is imported from langgraph.graph

    workflow.add_edge("fallback", END)
    # After fallback_node runs → always stop

    # Conditional edge — DYNAMIC routing
    workflow.add_conditional_edges(
        "grade",           # FROM this node
        should_generate,   # CALL this router function
        {                  # MAP return values to node names
            "generate": "generate",
            "fallback":  "fallback",
            "retrieve":  "retrieve",
        }
    )
    # After grade_node:
    # call should_generate(state)
    # if returns "generate" → go to generate node
    # if returns "fallback"  → go to fallback node
    # if returns "retrieve"  → go BACK to retrieve node
    # This creates the retry loop

    workflow.set_entry_point("retrieve")
    # Execution always starts at retrieve node
    # Like main() in Java — the entry point

    return workflow.compile()
    # compile() validates the graph structure
    # Checks all edges connect to real nodes
    # Returns a Runnable — can call .invoke() on it

# ============================================
# RUN THE AGENT
# ============================================

def run_agent(question: str, agent) -> str:
    """
    Run the compiled agent with one question.

    Args:
        question: user's question string
        agent:    compiled LangGraph agent

    Returns:
        final answer string
    """
    print(f"\nQuestion: '{question}'")
    print("=" * 50)

    initial_state = {
        "question":  question,
        "documents": [],
        "answer":    "",
        "grade":     "irrelevant",
        "attempts":  0,
    }
    # Initial state = starting values for all keys
    # question:  set by caller
    # documents: empty list — retrieve_node fills this
    # answer:    empty — generate or fallback fills this
    # grade:     start as irrelevant — grade_node updates
    # attempts:  0 — retrieve_node increments each call
    #
    # LangGraph merges partial updates from each node
    # into this state object as execution proceeds

    final_state = agent.invoke(initial_state)
    # .invoke() runs the full graph:
    # 1. Starts at entry point (retrieve)
    # 2. Each node runs and returns partial state update
    # 3. State is merged after each node
    # 4. Router decides next node
    # 5. Continues until END is reached
    # 6. Returns final merged state
    #
    # Java equivalent: workflowEngine.execute(context)

    print(f"\nAnswer: {final_state['answer']}")
    print(f"Attempts: {final_state['attempts']}")
    return final_state["answer"]
    # final_state has all state keys
    # answer = what generate or fallback put there
    # attempts = how many times retrieve ran

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    # Only runs when file executed directly
    # NOT when imported as a module
    # Java equivalent: public static void main()

    print("Building LangGraph agent...")
    agent = build_agent()
    print("Agent ready!")
    print()

    questions = [
        "What is the difference between V1 and V2?",
        "What is session state in Streamlit?",
        "How to make biryani?",
    ]

    for q in questions:
        run_agent(q, agent)
        print()