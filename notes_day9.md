Chunks:
Chunk count is NOT determined by page count.
Chunk count IS determined by text density.

Pages × text_per_page ÷ chunk_size = chunks

A 1-page PDF with 10,000 words
creates more chunks than a 50-page PDF
with mostly images and whitespace.

SmartDocs page (sparse):           CRAG page (dense):
┌─────────────────────┐            ┌─────────────────────┐
│  # Heading          │            │ Lorem ipsum dense   │
│                     │            │ text continues here │
│  [code block]       │            │ with references and │
│                     │            │ citations. Further  │
│  Some explanation   │            │ analysis shows that │
│                     │            │ the model performs  │
│  [diagram]          │            │ better on multiple  │
│                     │            │ benchmarks. Table 1 │
└─────────────────────┘            │ demonstrates this   │
                                   │ clearly. Moreover   │
~500 chars = 1-2 chunks            │ the ablation study  │
                                   └─────────────────────┘
                                   ~4000 chars = 8-10 chunks


 Feature              LangGraph          Google ADK
─────────────────    ──────────────     ──────────────
In-session           chat_history       InMemorySession
                     in TypedDict       Service

Cross-session        Mem0 / Zep         DatabaseSession
                     (you integrate)    Service (built in)

Auto-summarization   Manual code        VertexAiSession
                     (Day 17)           Service (automatic)

Semantic memory      Mem0 (Day 17+)     VertexAiMemory
                                        BankService

Control              Full — you         Less — Google
                     build everything   handles internals

Cloud lock-in        None               Google Cloud                                  

Google ADK Memory Services:

1. InMemorySessionService
   → Type 2 (in-session)
   → Python dict in RAM
   → Exact equivalent of your chat_history
   → Dies when process ends

2. DatabaseSessionService
   → Type 3 (cross-session)
   → Postgres or Firestore backend
   → Survives restarts
   → Equivalent of Mem0 in LangGraph world

3. VertexAiSessionService
   → Type 3 + automatic summarization
   → Google Cloud hosted
   → Auto-summarizes old conversation
   → Most production-ready option

4. VertexAiMemoryBankService (the one you forgot)
   → Type 3 + semantic search over memories
   → Stores facts not raw messages
   → "User prefers Hinglish" stored as a fact
   → Retrieved by similarity at query time
   → Most similar to Mem0

Day 9 built:
✅ chat_history in state (Type 2 memory)
✅ Last 4 messages sent to Claude
✅ memory_node saves Q&A after grounded answer
✅ History resets when script ends

Three gaps to understand now:
⬜ Memory window management
⬜ Memory summarization  
⬜ Cross-session persistence (Type 3)   

Strategy 1 — Fixed window (what you have):
Always send last N messages
Simple. Loses old context.

Strategy 2 — Summarization (Day 17):
"Summarize the conversation so far in 100 words"
Send summary + last 2 messages
Preserves old context compressed

Strategy 3 — Semantic retrieval (advanced):
Store all messages as vectors
Retrieve only relevant past messages
"You asked about GST 5 turns ago" found by similarity

Gap 2 — Memory Summarization
This is what Google ADK's VertexAiSessionService does automatically. LangGraph you build manually.
# When history gets too long — summarize it
def summarize_history(chat_history: list, llm) -> str:
    if len(chat_history) < 8:
        return ""  # no need yet

    old_messages = chat_history[:-4]  # everything except last 4
    recent       = chat_history[-4:]  # keep last 4 full

    summary_prompt = f"""Summarize this conversation 
    in 3 sentences preserving key facts discussed:
    {old_messages}"""

    summary = llm.invoke(summary_prompt)

    return f"Earlier conversation summary: {summary}\n\n"

# Then in generate_node:
summary      = summarize_history(chat_history, llm)
history_text = summary + format_recent(chat_history[-4:])

Gap 3 — Cross-Session Persistence

Day 9 (what you have):
Monday: chat_history grows through conversation
Monday: script ends
Tuesday: chat_history = []  ← starts fresh
         Agent remembers nothing from Monday

Day 17 (what you'll build with Mem0):
Monday: facts extracted from conversation
        "User is learning LangGraph"
        "User prefers Hinglish"
        "User asked about GST deadlines"
        → stored in database

Tuesday: agent loads facts from database
         "Welcome back Sumit. Last time we
          discussed LangGraph Day 9..."
         → persistent across days


Feature          ChromaDB        Pinecone
──────────────   ─────────────   ─────────────
Storage          Local disk      Cloud
Survives deploy  ❌ No           ✅ Yes
Setup            3 lines         3 lines
Version issues   Rare            Never
Cost             Free            Free tier
Scale            Single machine  Unlimited
Job mentions     Sometimes       Always
LangChain        ✅              ✅

Day 14: Docker Containerization
Day 15:Ragas
Day 16: Multimodal AI  (Multimodal = text + images together)
→ Send images to Claude
→ Claude reads scanned PDFs visually
→ Handles image-heavy documents
→ Tables, diagrams, handwritten notes
→ The biggest SmartDocs limitation solved
example:

# Text only (what you've done):
messages=[{
    "role": "user",
    "content": "What is CRAG?"
}]

# Multimodal (Day 16):
messages=[{
    "role": "user",
    "content": [
        {
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       base64_image_data,
            }
        },
        {
            "type": "text",
            "text": "Extract all text from this document"
        }
    ]
}]

# Images can't be sent as raw bytes over JSON
# Must be encoded as base64 string first
import base64

with open("page.png", "rb") as f:
    image_bytes = f.read()
    base64_str  = base64.standard_b64encode(
        image_bytes
    ).decode("utf-8")
# PDF to image conversion
pip install pdf2image pillow

# System dependency for pdf2image
sudo apt install poppler-utils -y 

Day 16 pipeline:
PDF → pdf2image → Claude Vision → text → answer
WORKS for ALL PDFs including scanned

Scanned government document:
PyPDFLoader:   0-5 words  ← fails completely
Claude Vision: 300+ words ← reads perfectly

That's when Vision becomes essential.

The Insight To Remember
Text PDF (SmartDocs guide):
→ PyPDFLoader = fine
→ Claude Vision = fine
→ Use PyPDFLoader (faster, cheaper)

Scanned PDF (government doc, court record):
→ PyPDFLoader = 0 words extracted ← fails
→ Claude Vision = full text ← works perfectly
→ Must use Vision

Mixed PDF (text + scanned pages):
→ Try PyPDFLoader first
→ If page has < 50 words → switch to Vision
→ Hybrid approach = best of both


User uploads PDF
      ↓
api.py /upload endpoint
      ↓
smart_extract(pdf_path)
      ↓
For each page:
  word_count >= 50? → PyPDFLoader ← fast, free
  word_count < 50?  → Claude Vision ← accurate
      ↓
Returns list of {page, text, method}
      ↓
Convert to LangChain Documents
      ↓
chunk_documents() ← same as before
      ↓
Store in Pinecone ← same as before
      ↓
Ready for Q&A


+++++++++++++++++++++++++++++

Day 17: Mem0 
Step 1 — After each conversation turn:
Claude reads the message and extracts facts:
"User is a CA firm owner"
"User prefers Hindi"
"User asked about GST 3 times"
"User is based in Lucknow"

Step 2 — Facts stored as vectors:
Each fact → embedding → stored in vector DB
(Mem0 manages this automatically)

Step 3 — Before each response:
Search stored facts relevant to current question
Inject facts into Claude's context
Claude answers with full user knowledge

pip install mem0ai