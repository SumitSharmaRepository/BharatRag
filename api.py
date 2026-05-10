# ============================================
# BharatRAG FastAPI Backend
# ============================================
# Exposes BharatRAG as a REST API.
# Any frontend, mobile app, or service
# can now call BharatRAG via HTTP.
#
# Java equivalent:
# This is your Spring Boot @RestController
# uvicorn = embedded Tomcat
# Pydantic = javax.validation
# ============================================

import os
import shutil
import tempfile
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# ── App setup ─────────────────────────────────────────
app = FastAPI(
    title       = "BharatRAG API",
    description = "AI-powered document Q&A for Indian professionals",
    version     = "1.0.0",
)
# FastAPI() creates the application
# title and description appear in auto-generated docs
# Like @SpringBootApplication in Java

# ── CORS middleware ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],  # allow all origins
    allow_methods     = ["*"],  # allow all HTTP methods
    allow_headers     = ["*"],  # allow all headers
)
# CORS = Cross-Origin Resource Sharing
# Allows your Streamlit frontend to call this API
# Without this: browser blocks cross-origin requests
# Java equivalent: @CrossOrigin annotation

# ── Config ─────────────────────────────────────────────
CHROMA_PATH     = "./chroma_db"
DATA_PATH       = "./data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")

# ── Global objects (loaded once at startup) ────────────
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

llm = ChatAnthropic(
    model             = "claude-sonnet-4-5",
    temperature       = 0,
    anthropic_api_key = ANTHROPIC_KEY,
)

def get_vectorstore():
    """Load existing ChromaDB."""
    return Chroma(
        embedding_function = embeddings,
        persist_directory  = CHROMA_PATH,
    )

def get_retriever(filter_dict=None):
    """Get retriever with optional document filter."""
    vectorstore = get_vectorstore()
    kwargs      = {"k": 3}
    if filter_dict:
        kwargs["filter"] = filter_dict
    return vectorstore.as_retriever(
        search_kwargs=kwargs
    )

# ============================================
# PYDANTIC MODELS — Request/Response schemas
# ============================================
# Pydantic validates incoming data automatically
# Wrong type = 422 Unprocessable Entity response
# Missing required field = 422 automatically
#
# Java equivalent:
# @RequestBody with @Valid and Bean Validation
# ============================================

class QueryRequest(BaseModel):
    question: str
    language: str  = "English"
    # Optional: filter to specific document
    doc_filter: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question":   "What is session state?",
                "language":   "English",
                "doc_filter": None,
            }
        }

class QueryResponse(BaseModel):
    answer:    str
    sources:   list[str]
    language:  str
    doc_count: int

class UploadResponse(BaseModel):
    filename:   str
    pages:      int
    chunks:     int
    total_chunks: int
    message:    str

class HealthResponse(BaseModel):
    status:      str
    chunks:      int
    model:       str
    version:     str

class DocumentsResponse(BaseModel):
    documents:   list[str]
    total_chunks: int

# ============================================
# ROUTES / ENDPOINTS
# ============================================
# Each function is an API endpoint.
# Decorator tells FastAPI the HTTP method and path.
#
# @app.get("/path")  → GET request
# @app.post("/path") → POST request
#
# Java equivalent:
# @GetMapping("/path")
# @PostMapping("/path")
# ============================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    GET /health
    Returns API status and basic stats.
    Use this to verify the API is running.
    """
    try:
        vs     = get_vectorstore()
        chunks = vs._collection.count()
    except Exception:
        chunks = 0

    return HealthResponse(
        status  = "healthy",
        chunks  = chunks,
        model   = "claude-sonnet-4-5",
        version = "1.0.0",
    )


@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    """
    GET /documents
    Lists all indexed documents in ChromaDB.
    Shows which PDFs have been uploaded.
    """
    try:
        vs      = get_vectorstore()
        results = vs._collection.get(
            include=["metadatas"]
        )
        # Extract unique document names
        doc_names = set()
        for meta in results["metadatas"]:
            if meta and "doc_name" in meta:
                doc_names.add(meta["doc_name"])

        return DocumentsResponse(
            documents    = sorted(list(doc_names)),
            total_chunks = vs._collection.count(),
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Error listing documents: {str(e)}"
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...)
):
    """
    POST /upload
    Upload a PDF file and index it to ChromaDB.

    Accepts: multipart/form-data with PDF file
    Returns: upload stats

    This replaces the manual copy-to-data-folder approach.
    Any client can upload PDFs via HTTP now.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code = 400,
            detail      = "Only PDF files accepted"
        )

    # Save uploaded file to temp location
    # tempfile ensures cleanup after processing
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        content = await file.read()
        # await = async file read
        # while file is being read, FastAPI handles other requests
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load and chunk the PDF
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()

        # Add filename to metadata
        for page in pages:
            page.metadata["doc_name"] = file.filename

        # Chunk it
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = 500,
            chunk_overlap = 100,
        )
        chunks = splitter.split_documents(pages)

        # Add to existing ChromaDB
        # (doesn't delete existing documents)
        vs = get_vectorstore()
        vs.add_documents(chunks)

        total = vs._collection.count()

        return UploadResponse(
            filename     = file.filename,
            pages        = len(pages),
            chunks       = len(chunks),
            total_chunks = total,
            message      = f"Successfully indexed {file.filename}",
        )

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Error processing PDF: {str(e)}"
        )
    finally:
        # Always clean up temp file
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    POST /query
    Ask a question about uploaded documents.

    This is the main endpoint.
    Retrieves relevant chunks and generates answer.

    Supports:
    - English, Hindi, Hinglish responses
    - Filter to specific document
    """
    if not request.question.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "Question cannot be empty"
        )

    # Build filter if specific document requested
    filter_dict = None
    if request.doc_filter:
        filter_dict = {"doc_name": request.doc_filter}

    # Retrieve relevant chunks
    retriever = get_retriever(filter_dict)
    docs      = retriever.invoke(request.question)

    if not docs:
        return QueryResponse(
            answer    = "I could not find relevant "
                       "information in the documents.",
            sources   = [],
            language  = request.language,
            doc_count = 0,
        )

    # Format chunks with citations
    doc_texts = []
    sources   = []

    for doc in docs:
        page     = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get("doc_name", "unknown")
        source   = f"{doc_name}, Page {int(page)+1}"
        sources.append(source)
        doc_texts.append(
            f"[{source}]\n{doc.page_content}"
        )

    context = "\n\n".join(doc_texts)

    # Language instructions
    lang_instruction = {
        "English":       "Answer in clear English.",
        "Hindi / हिंदी": "हिंदी में जवाब दें।",
        "Hinglish":      "Answer in Hinglish — natural "
                        "mix of Hindi and English.",
    }.get(request.language, "Answer in clear English.")

    # Generate answer
    prompt = f"""You are BharatRAG — an AI document assistant
for Indian professionals.

{lang_instruction}

Answer using ONLY the provided context.
If not found say: "I could not find this in the documents."
Always cite which document your answer comes from.

Context:
{context}

Question: {request.question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return QueryResponse(
        answer    = response.content,
        sources   = list(set(sources)),
        language  = request.language,
        doc_count = len(docs),
    )


@app.delete("/reset")
def reset_database():
    """
    DELETE /reset
    Clear all indexed documents from ChromaDB.
    Use with caution — cannot be undone.
    """
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            return {
                "message": "ChromaDB cleared successfully",
                "status":  "reset"
            }
        return {
            "message": "ChromaDB was already empty",
            "status":  "ok"
        }
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Error resetting: {str(e)}"
        )


# ============================================
# STARTUP EVENT
# ============================================
# Runs once when API starts.
# Good place to load models, check connections.
# Java equivalent: @PostConstruct
# ============================================

@app.on_event("startup")
async def startup_event():
    print("BharatRAG API starting...")
    print(f"Model: claude-sonnet-4-5")
    print(f"ChromaDB: {CHROMA_PATH}")
    try:
        vs = get_vectorstore()
        print(f"Chunks loaded: {vs._collection.count()}")
    except Exception:
        print("ChromaDB empty — upload PDFs via /upload")
    print("API ready at http://localhost:8000")
    print("Docs at http://localhost:8000/docs")