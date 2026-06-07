import os
import hashlib
import tempfile
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.retrieval.cache import get_cache

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from src.security import (
    validate_question,
    validate_filename,
    get_pii_detector,
)

from fastapi.responses import StreamingResponse
import json

ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "bharatrag")

app = FastAPI(
    title       = "BharatRAG API",
    description = "AI-powered document Q&A for Indian professionals",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "https://bharatrag.vercel.app",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "*",
    ],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
    allow_credentials = True,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Low-level Pinecone helpers ─────────────────────────────

_pinecone_index = None

def _get_pinecone_index():
    """Singleton Pinecone index — created once, reused forever."""
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone as PineconeClient
        pc = PineconeClient(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index

_DUMMY_VEC = [0.0] * 1024

def _get_chunks_for_doc(index, user_id: str, doc_name: str) -> list[str]:
    """Return all vector IDs for a user's document (any archived state)."""
    result = index.query(
        vector=_DUMMY_VEC,
        top_k=500,
        include_metadata=True,
        filter={
            "user_id":  {"$eq": user_id},
            "doc_name": {"$eq": doc_name},
        },
    )
    return [m["id"] for m in result.get("matches", [])]

def _update_chunks_archived(index, chunk_ids: list[str], archived: bool) -> None:
    """Partial-update the archived flag on a list of vector IDs."""
    for chunk_id in chunk_ids:
        index.update(id=chunk_id, set_metadata={"archived": archived})


# ── Higher-level helpers ───────────────────────────────────

def get_embeddings():
    from src.embeddings import get_embeddings as _get
    return _get()

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        from langchain_anthropic import ChatAnthropic
        _llm = ChatAnthropic(
            model             = "claude-sonnet-4-5",
            temperature       = 0,
            anthropic_api_key = ANTHROPIC_KEY,
        )
    return _llm

def get_vectorstore():
    from langchain_pinecone import PineconeVectorStore
    return PineconeVectorStore(
        index_name = PINECONE_INDEX,
        embedding  = get_embeddings(),
    )

def get_retriever(filter_dict: dict = None):
    kwargs = {"k": 3}
    if filter_dict:
        kwargs["filter"] = filter_dict
    return get_vectorstore().as_retriever(search_kwargs=kwargs)

def get_pinecone_chunk_count() -> int:
    try:
        stats = _get_pinecone_index().describe_index_stats()
        return stats.get("total_vector_count", 0)
    except Exception:
        return 0

def get_pinecone_documents(user_id: str) -> dict:
    """Return {active, archived, total_chunks} for a user — one index, three calls."""
    try:
        index = _get_pinecone_index()

        active_result = index.query(
            vector=_DUMMY_VEC, top_k=100, include_metadata=True,
            filter={
                "user_id":  {"$eq": user_id},
                "archived": {"$ne": True},
            },
        )
        active_docs = {
            m["metadata"]["doc_name"]
            for m in active_result.get("matches", [])
            if "doc_name" in m.get("metadata", {})
        }

        archived_result = index.query(
            vector=_DUMMY_VEC, top_k=100, include_metadata=True,
            filter={
                "user_id":  {"$eq": user_id},
                "archived": {"$eq": True},
            },
        )
        archived_docs = {
            m["metadata"]["doc_name"]
            for m in archived_result.get("matches", [])
            if "doc_name" in m.get("metadata", {})
        } - active_docs

        total = index.describe_index_stats().get("total_vector_count", 0)

        return {
            "active":        sorted(active_docs),
            "archived":      sorted(archived_docs),
            "total_chunks":  total,
        }
    except Exception:
        return {"active": [], "archived": [], "total_chunks": 0}


# ── Pydantic models ────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    language:   str           = "English"
    doc_filter: Optional[str] = None
    user_id:    str           = "default_user"

class QueryResponse(BaseModel):
    answer:     str
    sources:    list[str]
    language:   str
    doc_count:  int
    agent_used: str = "RAGPipeline"

class UploadResponse(BaseModel):
    filename:     str
    pages:        int
    chunks:       int
    total_chunks: int
    message:      str
    status:       str = "new"   # "new" | "skipped" | "restored"

class HealthResponse(BaseModel):
    status:  str
    chunks:  int
    model:   str
    version: str

class DocumentsResponse(BaseModel):
    active:       list[str]
    archived:     list[str]
    total_chunks: int


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    chunks = get_pinecone_chunk_count()
    return HealthResponse(
        status  = "healthy",
        chunks  = chunks,
        model   = "claude-sonnet-4-5",
        version = "1.0.0",
    )


@app.get("/documents", response_model=DocumentsResponse)
@limiter.limit("30/minute")
def list_documents(request: Request, user_id: str = "default_user"):
    try:
        docs = get_pinecone_documents(user_id)
        return DocumentsResponse(
            active       = docs["active"],
            archived     = docs["archived"],
            total_chunks = docs["total_chunks"],
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("5/hour")
async def upload_document(
    request: Request,
    file:    UploadFile = File(...),
    user_id: str        = Form("default_user"),
):
    # ── Filename validation ────────────────────────
    fname_check = validate_filename(file.filename)
    if not fname_check["valid"]:
        raise HTTPException(400, fname_check["reason"])

    # ── Read bytes + fingerprint ───────────────────
    file_bytes = await file.read()
    file_hash  = hashlib.sha256(file_bytes).hexdigest()

    # ── Dedup check ────────────────────────────────
    index = _get_pinecone_index()
    dup   = index.query(
        vector=_DUMMY_VEC,
        top_k=1,
        include_metadata=True,
        filter={
            "user_id":   {"$eq": user_id},
            "file_hash": {"$eq": file_hash},
        },
    )
    matches = dup.get("matches", [])

    if matches:
        meta        = matches[0].get("metadata", {})
        is_archived = meta.get("archived", False)

        if not is_archived:
            return UploadResponse(
                filename=file.filename, pages=0, chunks=0,
                total_chunks=get_pinecone_chunk_count(),
                message="already indexed, skipped",
                status="skipped",
            )

        # Restore all archived chunks for this file_hash
        all_chunks = index.query(
            vector=_DUMMY_VEC, top_k=500, include_metadata=True,
            filter={
                "user_id":   {"$eq": user_id},
                "file_hash": {"$eq": file_hash},
            },
        )
        chunk_ids = [m["id"] for m in all_chunks.get("matches", [])]
        _update_chunks_archived(index, chunk_ids, False)
        return UploadResponse(
            filename=file.filename, pages=0, chunks=len(chunk_ids),
            total_chunks=get_pinecone_chunk_count(),
            message="restored, free",
            status="restored",
        )

    # ── New file — write to temp ───────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()

        for page in pages:
            page.metadata["doc_name"] = file.filename

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = splitter.split_documents(pages)

        # ── PII detection + redaction ──────────────
        pii       = get_pii_detector()
        pii_found = set()
        clean_chunks = []

        for chunk in chunks:
            detected = pii.detect(chunk.page_content)
            if detected:
                pii_found.update(detected)
                chunk.page_content = pii.redact(chunk.page_content)
            clean_chunks.append(chunk)

        if pii_found:
            print(f"  [security] PII redacted: {pii_found}")

        # ── Stamp metadata ─────────────────────────
        uploaded_at = datetime.now(timezone.utc).isoformat()
        for chunk in clean_chunks:
            chunk.metadata["user_id"]     = user_id
            chunk.metadata["file_hash"]   = file_hash
            chunk.metadata["uploaded_at"] = uploaded_at
            chunk.metadata["archived"]    = False

        # ── Store in Pinecone ──────────────────────
        vs = get_vectorstore()
        vs.add_documents(clean_chunks)

        return UploadResponse(
            filename=file.filename,
            pages=len(pages),
            chunks=len(clean_chunks),
            total_chunks=get_pinecone_chunk_count(),
            message=f"Indexed {file.filename}",
            status="new",
        )

    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_documents(
    request: Request,
    body:    QueryRequest,
):
    # ── Input validation ───────────────────────────
    validation = validate_question(body.question)
    if not validation["valid"]:
        raise HTTPException(400, validation["reason"])

    question = validation["cleaned"]

    # ── Check cache ────────────────────────────────
    cache  = get_cache()
    cached = cache.get(question)

    if cached:
        print(f"  [api] Cache HIT")
        return QueryResponse(
            answer     = cached["answer"],
            sources    = cached["sources"],
            language   = body.language,
            doc_count  = len(cached["sources"]),
            agent_used = "Cache",
        )

    # ── Build filter (user isolation + archive) ────
    filter_dict: dict = {
        "user_id":  {"$eq": body.user_id},
        "archived": {"$ne": True},
    }
    if body.doc_filter:
        filter_dict["doc_name"] = {"$eq": body.doc_filter}

    # ── Retrieve ───────────────────────────────────
    retriever = get_retriever(filter_dict)
    docs      = retriever.invoke(question)

    if not docs:
        return QueryResponse(
            answer     = "I could not find relevant information.",
            sources    = [],
            language   = body.language,
            doc_count  = 0,
            agent_used = "RAGPipeline",
        )

    # ── Format context ─────────────────────────────
    doc_texts = []
    sources   = []
    for doc in docs:
        page     = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get("doc_name", "unknown")
        source   = f"{doc_name}, Page {int(page)+1}"
        sources.append(source)
        doc_texts.append(f"[{source}]\n{doc.page_content}")

    context = "\n\n".join(doc_texts)

    # ── Language instruction ───────────────────────
    lang_instruction = {
        "English":       "Answer in clear English.",
        "Hindi / हिंदी": "हिंदी में जवाब दें।",
        "Hinglish":      "Answer in Hinglish naturally.",
        "Arabic / عربي": "أجب باللغة العربية بوضوح.",
    }.get(body.language, "Answer in clear English.")

    # ── Generate ───────────────────────────────────
    prompt = f"""You are BharatRAG — an AI document assistant
for Indian professionals.

{lang_instruction}

Answer using ONLY the provided context.
If not found: "I could not find this in the documents."
Always cite which document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

    from langchain_core.messages import HumanMessage
    response       = get_llm().invoke([HumanMessage(content=prompt)])
    answer         = response.content
    unique_sources = list(set(sources))

    cache.set(question=question, answer=answer, sources=unique_sources)

    return QueryResponse(
        answer     = answer,
        sources    = unique_sources,
        language   = body.language,
        doc_count  = len(docs),
        agent_used = "RAGPipeline",
    )


@app.get("/stream")
async def stream_query(
    question:   str = "",
    language:   str = "English",
    user_id:    str = "default_user",
    doc_filter: str = "",
):
    """GET /stream — Server-Sent Events streaming response."""
    if not question.strip():
        async def error_stream():
            yield 'data: {"error": "Question required"}\n\n'
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def generate():
        try:
            yield ": ping\n\n"

            filter_dict: dict = {
                "user_id":  {"$eq": user_id},
                "archived": {"$ne": True},
            }
            if doc_filter:
                filter_dict["doc_name"] = {"$eq": doc_filter}

            retriever = get_retriever(filter_dict)
            docs      = retriever.invoke(question)

            if not docs:
                yield (
                    "data: " +
                    json.dumps({"chunk": "I could not find relevant information."}) +
                    "\n\n"
                )
                yield "data: [DONE]\n\n"
                return

            doc_texts = []
            sources   = []
            for doc in docs:
                page     = doc.metadata.get("page", "?")
                doc_name = doc.metadata.get("doc_name", "unknown")
                source   = f"{doc_name}, Page {int(page)+1}"
                sources.append(source)
                doc_texts.append(f"[{source}]\n{doc.page_content}")

            context = "\n\n".join(doc_texts)

            lang_instruction = {
                "English":       "Answer in clear English.",
                "Hindi / हिंदी": "हिंदी में जवाब दें।",
                "Hinglish":      "Answer in Hinglish naturally.",
                "Arabic / عربي": "أجب باللغة العربية بوضوح.",
            }.get(language, "Answer in clear English.")

            prompt = f"""You are BharatRAG — an AI document
assistant for Indian professionals.

{lang_instruction}

Answer using ONLY the provided context.
If not found say:
"I could not find this in the documents."
Always cite which document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

            import anthropic
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

            with anthropic_client.messages.stream(
                model      = "claude-sonnet-4-5",
                max_tokens = 1024,
                messages   = [{"role": "user", "content": prompt}],
            ) as stream:
                for text_chunk in stream.text_stream:
                    yield "data: " + json.dumps({"chunk": text_chunk}) + "\n\n"

            yield ": padding\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.delete("/documents/{doc_name}")
@limiter.limit("10/hour")
async def delete_document(
    request:  Request,
    doc_name: str,
    user_id:  str = "default_user",
    mode:     str = "archive",   # "archive" | "permanent"
):
    try:
        index = _get_pinecone_index()

        if mode == "permanent":
            index.delete(
                filter={
                    "user_id":  {"$eq": user_id},
                    "doc_name": {"$eq": doc_name},
                }
            )
            get_cache().clear()
            return {"message": f"Permanently deleted {doc_name}", "status": "deleted"}

        # Archive: flip archived=True on all chunks
        chunk_ids = _get_chunks_for_doc(index, user_id, doc_name)
        _update_chunks_archived(index, chunk_ids, True)
        return {"message": f"Archived {doc_name}", "status": "archived"}

    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.post("/documents/{doc_name}/restore")
@limiter.limit("10/hour")
async def restore_document(
    request:  Request,
    doc_name: str,
    user_id:  str = "default_user",
):
    try:
        index     = _get_pinecone_index()
        chunk_ids = _get_chunks_for_doc(index, user_id, doc_name)
        _update_chunks_archived(index, chunk_ids, False)
        return {"message": f"Restored {doc_name}", "status": "restored"}
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.delete("/reset")
def reset_database():
    try:
        index = _get_pinecone_index()
        index.delete(delete_all=True)
        return {"message": "Pinecone cleared", "status": "reset"}
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    print("BharatRAG API starting...", flush=True)
    print(f"Model:     claude-sonnet-4-5", flush=True)
    print(f"Vector DB: Pinecone ({PINECONE_INDEX})", flush=True)
    print("API ready — embeddings load on first request", flush=True)
    print(f"Docs at /docs", flush=True)


@app.get("/cache/stats")
def cache_stats():
    return get_cache().stats()

@app.delete("/cache/clear")
def cache_clear():
    get_cache().clear()
    return {"message": "Cache cleared"}
