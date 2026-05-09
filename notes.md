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


Modular Project Structure PROD level
PRODUCTION PYTHON PROJECT STRUCTURE
====================================

bharatrag/
│
├── main.py              # entry point — ties modules together
├── config.py            # ALL settings in one place (chunk_size, model names, paths)
├── requirements.txt     # dependencies — pip install -r requirements.txt
├── .env                 # API keys — NEVER commit to git
├── .gitignore           # excludes: venv/, .env, chroma_db/, __pycache__/
├── pyrightconfig.json   # VS Code type checking config
├── CLAUDE.md            # rules for Claude Code AI assistant
├── Dockerfile           # container build for deployment
│
├── src/                 # ALL application code lives here
│   ├── __init__.py      # marks folder as Python package — enables imports
│   ├── loader.py        # PDF loading + chunking ONLY
│   ├── vectorstore.py   # ChromaDB management ONLY
│   ├── chain.py         # LangChain RAG chain ONLY
│   └── prompts.py       # prompt templates ONLY
│
├── tests/               # mirrors src/ structure
│   ├── conftest.py      # shared pytest fixtures
│   ├── pytest.ini       # test configuration
│   ├── test_loader.py   # tests for loader.py
│   ├── test_vectorstore.py
│   └── test_chain.py
│
├── data/                # PDF source documents
│   └── *.pdf
│
├── docs/                # project documentation
│   ├── README.md
│   ├── SDD.md           # Software Design Document
│   └── CHANGELOG.md
│
├── Day1TODay4_Learning/ # learning files — NOT production code
│   └── day1_.py → day6_.py
│
├── venv/                # in .gitignore — never commit
└── chroma_db/           # in .gitignore — regenerated from PDFs

RULES:
- one file = one responsibility (single responsibility principle)
- config.py controls everything — change settings in ONE place
- main.py has no business logic — only imports and orchestration
- test files mirror src/ structure exactly
- run tests with: pytest tests/
- Java equivalent: src/main/java + src/test/java in Maven

LANGGRAPH PROJECT STRUCTURE — PRODUCTION
==========================================

bharatrag/
│
├── main.py              # entry point — builds and runs the agent
├── config.py            # chunk_size, model names, k value, paths
├── requirements.txt     # langchain, langgraph, chromadb, anthropic
├── .env                 # ANTHROPIC_API_KEY — never commit
│
├── src/
│   │
│   ├── agents/          # LangGraph graphs live here
│   │   ├── rag_agent.py # StateGraph build + compile()
│   │   ├── state.py     # TypedDict state definition
│   │   └── router.py    # conditional edge functions
│   │
│   ├── nodes/           # one file per node function
│   │   ├── retrieve_node.py  # searches ChromaDB
│   │   ├── grade_node.py     # relevance grading
│   │   ├── generate_node.py  # Claude answer generation
│   │   └── fallback_node.py  # honest "not found" answer
│   │
│   ├── tools/           # LLM callable tools (Day 7+)
│   │   ├── search_tool.py
│   │   └── calculator_tool.py
│   │
│   ├── retrieval/       # vector store layer
│   │   ├── vectorstore.py
│   │   └── embeddings.py
│   │
│   └── prompts/         # all prompt templates
│       ├── rag_prompt.py
│       └── hindi_prompt.py
│
├── tests/
│   ├── test_nodes.py    # unit test each node independently
│   ├── test_agent.py    # end-to-end agent test
│   └── test_retrieval.py
│
├── api.py               # FastAPI routes (Day 12)
├── app.py               # Streamlit UI
├── Dockerfile           # container deployment
├── CLAUDE.md            # rules for Claude Code
└── .gitignore           # venv/, .env, chroma_db/

KEY CONCEPTS:
─────────────
state.py     → TypedDict = shared memory across all nodes
               question, documents, answer, grade, attempts

rag_agent.py → workflow = StateGraph(RAGState)
               workflow.add_node("retrieve", retrieve_node)
               workflow.add_edge("retrieve", "grade")
               workflow.add_conditional_edges("grade", router)
               agent = workflow.compile()

router.py    → def should_generate(state) -> str:
               reads state["grade"] and state["attempts"]
               returns "generate" / "fallback" / "retrieve"

nodes/       → each node reads state, does ONE job, returns partial state
               retrieve → grade → generate/fallback

WHAT YOU HAVE NOW vs FULL STRUCTURE:
─────────────────────────────────────
Current:  everything in one file (langgraph_intro.py)
Target:   split across agents/ nodes/ retrieval/ prompts/
When:     Day 7-10 we refactor into this structure

DAY 8
LLM-as-a-Judge

retrieve
   ↓
grade
   ↓
rewrite if needed
   ↓
generate
   ↓
hallucination_check
   ↓
regenerate if needed

                ┌──────────┐
                │ retrieve │
                └────┬─────┘
                     ↓
                ┌──────────┐
                │  grade   │
                └────┬─────┘
                     │
      ┌──────────────┼──────────────┐
      ↓              ↓              ↓
┌──────────┐   ┌──────────┐   ┌──────────┐
│ generate │   │ rewrite  │   │ fallback │
└────┬─────┘   └────┬─────┘   └────┬─────┘
     ↓              ↓              ↓
┌──────────────┐  retrieve        END
│ hallucination│
│    check     │
└────┬─────────┘
     │
 ┌───┴───────────┐
 ↓               ↓
END         regenerate
                 ↓
             generate

QnA

Q1 — grade_node vs hallucination_check_node
grade_node:
Checks: are the RETRIEVED CHUNKS relevant to the question?
Runs:   BEFORE generation
Input:  chunks from ChromaDB
Output: relevant/irrelevant

hallucination_check_node:
Checks: is the GENERATED ANSWER supported by the chunks?
Runs:   AFTER generation
Input:  Claude's answer + chunks
Output: grounded/hallucinated

Q2 — Stricter prompt on second attempt

First attempt:  normal prompt → Claude answers naturally
                → hallucination checker catches it
                → Claude added outside knowledge

Second attempt: STRICT prompt → "Do NOT add any outside
                knowledge. Only exact information from context."
                → Forces Claude to stay within chunks only
                → More likely to pass hallucination check

Q3 — gen_attempts < 2 check
 WITH gen_attempts < 2 check:
hallucinated → regenerate (attempt 2)
hallucinated again → return anyway (stop)
Maximum 2 generation attempts

WITHOUT gen_attempts < 2 check:
hallucinated → regenerate → hallucinated → regenerate
→ hallucinated → regenerate → infinite loop
Agent never returns an answer
Application hangs forever

Q4 — LLM-as-judge definition

LLM-as-judge = using a language model to evaluate
               the output of another language model
               (or itself with a different prompt)
               instead of using human evaluation.
 it's specifically about EVALUATION replacing human review. 5/10

Q5 — Biryani Generation attempts: 0
Biryani question path:
retrieve → grade=irrelevant → rewrite
        → retrieve → grade=irrelevant → FALLBACK

Fallback runs INSTEAD of generate.
generate_node NEVER ran.
hallucination_check NEVER ran.

generation_attempts starts at 0.
Nobody incremented it.
So it stayed 0.

This proves:
→ Fallback bypasses generation entirely
→ No hallucination possible in fallback
→ Fallback is always "grounded" by definition
   (it's a hardcoded honest message)

GO BACK ND RE READ 
1. The STATE — what fields exist and who updates them
2. The ROUTERS — what each return value means
3. The FLOW — draw it on paper

retrieve → grade → [router]
                 → relevant  → generate → hallucination_check
                                        → grounded    → END
                                        → hallucinated → generate
                 → irrelevant attempt 1 → rewrite → retrieve
                 → irrelevant attempt 2 → fallback → END   