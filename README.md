# BharatRAG

**AI-powered document intelligence for Indian professionals.**
Ask questions from your PDFs in English, Hindi, Hinglish, or Arabic. Built with LangGraph, Claude API, Pinecone, and FastAPI.

---

## Live Demo

> SmartDocs AI (V1 product built on the same stack): **[asksmartdocs-ai.streamlit.app](https://asksmartdocs-ai.streamlit.app)**

---

## What Is BharatRAG

BharatRAG is a production-grade Retrieval Augmented Generation (RAG) system with a multi-agent architecture. Upload any PDF — text or scanned — and ask questions about it in your language of choice. The system routes your question to a specialist agent, retrieves relevant chunks from a cloud vector database, checks for hallucinations, and returns a grounded, cited answer.

Built as a 30-day learning project to go from zero to production GenAI engineering.

---

## Key Features

- **Multi-agent routing** — Supervisor classifies questions and routes to specialist agents (Tech, Research, Logistics, General)
- **Self-correcting RAG** — Query rewriting, relevance grading, and hallucination detection before every answer
- **Persistent memory** — Mem0 remembers user preferences across sessions (language, domain, past questions)
- **Multilingual** — English, Hindi (हिंदी), Hinglish, Arabic (عربي)
- **Scanned PDF support** — Claude Vision OCR for image-based documents
- **Production infrastructure** — FastAPI backend, Pinecone vector DB, LangSmith observability, RAGAS evaluation
- **React frontend** — Professional chat UI with dark mode, file upload, language selector, source citations

---

## Architecture

```
User Question
      │
      ▼
Mem0 — fetch user facts (cross-session memory)
      │
      ▼
Supervisor — classify domain
      │
   ┌──┴──────────────────┐
   ▼          ▼          ▼          ▼
TechAgent  ResearchAgent  LogisticsAgent  GeneralAgent
   │          │              │               │
   └──────────┴──────────────┴───────────────┘
                    │
                    ▼
          Pinecone semantic search
          (metadata filter per specialist)
                    │
                    ▼
          grade_node → query_rewrite → generate
                    │
                    ▼
          hallucination_check
                    │
                    ▼
          Language formatter (EN/HI/Hinglish/AR)
                    │
                    ▼
          Mem0 — save new user facts
                    │
                    ▼
              Answer + citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude Sonnet (Anthropic API) |
| Agent framework | LangGraph |
| Vector database | Pinecone |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384 dims) |
| Persistent memory | Mem0 + Qdrant (local) |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + Tailwind CSS (Vite) |
| OCR / Vision | Claude Vision API + pdf2image |
| Observability | LangSmith |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) |
| Containerisation | Docker + docker-compose |

---

## Project Structure

```
bharatrag/
├── api.py                        # FastAPI backend (5 endpoints)
├── config.py                     # Centralised settings
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── test_day18.py                 # Integration tests
│
├── src/
│   ├── agents/
│   │   ├── state.py              # RAGState + MultiAgentState TypedDicts
│   │   ├── supervisor.py         # Supervisor + memory_retrieve_node
│   │   └── graph.py              # build_multi_agent()
│   │
│   ├── nodes/                    # Individual LangGraph nodes
│   │   ├── retrieve.py
│   │   ├── grade.py
│   │   ├── rewrite.py
│   │   ├── hallucination.py
│   │   └── memory_node.py
│   │
│   ├── specialists/              # Domain specialist agents
│   │   ├── base.py               # Shared LLM + embeddings + retriever
│   │   ├── tech_agent.py         # SmartDocs / Python / AI tools
│   │   ├── research_agent.py     # Research papers / CRAG / RAG systems
│   │   ├── logistics_agent.py    # Invoices / challans / purchase orders
│   │   └── general_agent.py     # Fallback — all documents
│   │
│   ├── memory/
│   │   ├── persistent.py         # Mem0 cross-session memory (Type 3)
│   │   └── session.py            # chat_history in-session memory (Type 2)
│   │
│   └── vision/
│       └── ocr.py                # Claude Vision for scanned PDFs
│
├── eval/
│   ├── ragas_eval.py             # RAGAS 4-metric evaluation
│   └── langsmith_eval.py         # LangSmith experiment runner
│
└── frontend/                     # React + Tailwind UI
    ├── src/
    │   ├── App.jsx
    │   ├── api/bharatrag.js      # All FastAPI calls
    │   └── components/
    │       ├── Header.jsx
    │       ├── Sidebar.jsx
    │       ├── Chat.jsx
    │       ├── Message.jsx
    │       └── ThemeToggle.jsx
    └── vite.config.js
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API status + chunk count |
| GET | `/documents` | List indexed PDFs |
| POST | `/upload` | Upload and index a PDF |
| POST | `/query` | Ask a question (with language + user_id) |
| DELETE | `/reset` | Clear all indexed vectors |

Full interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- Anthropic API key
- Pinecone API key (free tier)
- LangSmith API key (free tier, optional)

### Backend Setup

```bash
# Clone the repo
git clone https://github.com/SumitSharmaRepository/BharatRag.git
cd BharatRag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Start the API
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### Environment Variables

```
ANTHROPIC_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX=bharatrag
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=bharatrag
```

---

## Memory System

BharatRAG uses a four-layer memory architecture:

| Type | Implementation | Survives |
|---|---|---|
| Type 1 — In-context | PDF chunks in prompt | Single API call |
| Type 2 — In-session | chat_history in LangGraph state | One conversation |
| Type 3 — Cross-session | Mem0 + Qdrant vector store | Forever |
| Type 4 — External KB | Pinecone document chunks | Forever |

---

## Evaluation Results

Evaluated using RAGAS on a 10-question golden dataset:

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.92 | Answers grounded in retrieved context |
| Answer Relevancy | 0.87 | Answer addresses the question |
| Context Precision | 0.76 | Retrieved chunks are relevant |
| Context Recall | 0.71 | Retrieval finds all needed content |

LangSmith experiment baseline: **86.7% average score**.

---

## LangGraph Agent Patterns

Each pattern from the literature is implemented and tested:

| Pattern | Day | Description |
|---|---|---|
| Basic RAG agent | Day 6 | retrieve → grade → generate → fallback |
| Reflexion | Day 7 | query_rewrite on retrieval failure |
| LLM-as-judge | Day 8 | hallucination_check after every generation |
| Conversation memory | Day 9 | chat_history persists within session |
| Supervisor multi-agent | Day 10 | Domain routing to specialists |
| Persistent memory | Day 17 | Mem0 cross-session user facts |
| Production multi-agent | Day 18 | All patterns combined |

---

## Multilingual Support

BharatRAG is built for the Indian market and the Gulf Indian diaspora:

```python
LANG_INSTRUCTIONS = {
    "English":       "Answer in clear English.",
    "Hindi / हिंदी": "हिंदी में जवाब दें।",
    "Hinglish":      "Answer in Hinglish naturally.",
    "Arabic / عربي": "أجب باللغة العربية بوضوح.",
}
```

---

## Roadmap

- [x] Self-correcting RAG with LangGraph
- [x] Multi-agent supervisor pattern
- [x] Persistent memory with Mem0
- [x] Multilingual (EN / HI / Hinglish / AR)
- [x] Scanned PDF support via Claude Vision
- [x] Production API with FastAPI
- [x] Cloud vector storage with Pinecone
- [x] LangSmith observability
- [x] RAGAS evaluation
- [x] React frontend with dark mode
- [ ] Hybrid search (BM25 + dense) — Day 22
- [ ] Streaming responses — Day 23
- [ ] User authentication — Day 24
- [ ] WhatsApp integration — Day 26
- [ ] Railway + Vercel deployment — Day 20

---

## Built By

**Sumit Sharma**

- LinkedIn: [linkedin.com/in/sumit-sharma-dev](https://linkedin.com/in/sumit-sharma-dev)
- GitHub: [github.com/SumitSharmaRepository](https://github.com/SumitSharmaRepository)
- SmartDocs AI: [asksmartdocs-ai.streamlit.app](https://asksmartdocs-ai.streamlit.app)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
