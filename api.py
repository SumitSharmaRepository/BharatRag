import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from pinecone import Pinecone as PineconeClient

from src.retrieval.cache import get_cache

#Security imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from src.security import (
    validate_question,
    validate_filename,
    get_pii_detector,
)

#streaming imports
from fastapi.responses import StreamingResponse
import anthropic
import json

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
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
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#Add Rate limiter after app creation to avoid circular imports with security.py
# ── Rate limiter ──────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)
# 10 requests per minute per IP for /query
# 5 uploads per hour per IP for /upload
# Prevents cost abuse and API hammering

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

llm = ChatAnthropic(
    model             = "claude-sonnet-4-5",
    temperature       = 0,
    anthropic_api_key = ANTHROPIC_KEY,
)

def get_vectorstore():
    return PineconeVectorStore(
        index_name = PINECONE_INDEX,
        embedding  = embeddings,
    )

def get_retriever(filter_dict: dict = None):
    kwargs = {"k": 3}
    if filter_dict:
        kwargs["filter"] = filter_dict
    return get_vectorstore().as_retriever(
        search_kwargs=kwargs
    )

def get_pinecone_chunk_count() -> int:
    try:
        pc    = PineconeClient(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()
        return stats.get("total_vector_count", 0)
    except Exception:
        return 0

def get_pinecone_documents() -> list:
    try:
        pc    = PineconeClient(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        dummy  = [0.0] * 384
        result = index.query(
            vector           = dummy,
            top_k            = 100,
            include_metadata = True,
        )
        doc_names = set()
        for match in result.get("matches", []):
            meta = match.get("metadata", {})
            if "doc_name" in meta:
                doc_names.add(meta["doc_name"])
        return sorted(list(doc_names))
    except Exception:
        return []


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

class HealthResponse(BaseModel):
    status:  str
    chunks:  int
    model:   str
    version: str

class DocumentsResponse(BaseModel):
    documents:    list[str]
    total_chunks: int


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
def list_documents():
    try:
        docs   = get_pinecone_documents()
        chunks = get_pinecone_chunk_count()
        return DocumentsResponse(
            documents    = docs,
            total_chunks = chunks,
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("5/hour")
async def upload_document(
    request: Request,
    file:    UploadFile = File(...),
):
    # ── Filename validation ────────────────────────
    fname_check = validate_filename(file.filename)
    if not fname_check["valid"]:
        raise HTTPException(400, fname_check["reason"])

    # ── Save to temp file ──────────────────────────
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # ── Load PDF ───────────────────────────────
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()

        for page in pages:
            page.metadata["doc_name"] = file.filename

        # ── Chunk ──────────────────────────────────
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
                chunk.page_content = pii.redact(
                    chunk.page_content
                )
            clean_chunks.append(chunk)

        if pii_found:
            print(
                f"  [security] PII redacted: {pii_found}"
            )

        # ── Store in Pinecone ──────────────────────
        vs = get_vectorstore()
        vs.add_documents(clean_chunks)

        total = get_pinecone_chunk_count()

        return UploadResponse(
            filename     = file.filename,
            pages        = len(pages),
            chunks       = len(clean_chunks),
            total_chunks = total,
            message      = f"Indexed {file.filename}",
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

    # ── Build filter ───────────────────────────────
    filter_dict = None
    if body.doc_filter:
        filter_dict = {"doc_name": body.doc_filter}

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
        doc_texts.append(
            f"[{source}]\n{doc.page_content}"
        )

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

    response       = llm.invoke([HumanMessage(content=prompt)])
    answer         = response.content
    unique_sources = list(set(sources))

    # ── Store in cache ─────────────────────────────
    cache.set(
        question = question,
        answer   = answer,
        sources  = unique_sources,
    )

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
    """
    GET /stream
    Streams Claude's response word by word.
    Uses Server-Sent Events (SSE).

    Client connects and keeps connection open.
    Server sends chunks as Claude generates them.
    Client displays each chunk immediately.

    Why GET not POST:
    EventSource API (browser built-in) only
    supports GET requests. So we pass params
    in query string instead of request body.
    """
    if not question.strip():
        async def error_stream():
            yield "data: {\"error\": \"Question required\"}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )

    async def generate():
        """
        Generator function that yields SSE events.
        Each yield sends one chunk to the browser.

        SSE format:
        data: {"chunk": "hello"}\n\n
        data: {"chunk": " world"}\n\n
        data: [DONE]\n\n
        """
        try:
            # Send ping immediately so browser
            # doesn't timeout before retrieval
            yield ": ping\n\n"
            
            # Step 1 — Retrieve relevant chunks
            filter_dict = None
            if doc_filter:
                filter_dict = {"doc_name": doc_filter}

            retriever = get_retriever(filter_dict)
            docs      = retriever.invoke(question)

            if not docs:
                yield (
                    "data: " +
                    json.dumps({
                        "chunk": "I could not find "
                                 "relevant information."
                    }) + "\n\n"
                )
                yield "data: [DONE]\n\n"
                return

            # Format context
            doc_texts = []
            sources   = []
            for doc in docs:
                page     = doc.metadata.get("page", "?")
                doc_name = doc.metadata.get(
                    "doc_name", "unknown"
                )
                source   = f"{doc_name}, Page {int(page)+1}"
                sources.append(source)
                doc_texts.append(
                    f"[{source}]\n{doc.page_content}"
                )

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

            # Step 2 — Stream Claude response
            # Use Anthropic client directly for streaming
            anthropic_client = anthropic.Anthropic(
                api_key=ANTHROPIC_KEY
            )

            with anthropic_client.messages.stream(
                model      = "claude-sonnet-4-5",
                max_tokens = 1024,
                messages   = [
                    {"role": "user", "content": prompt}
                ],
            ) as stream:
                for text_chunk in stream.text_stream:
                    # Send each chunk as SSE event
                    yield (
                        "data: " +
                        json.dumps({"chunk": text_chunk}) +
                        "\n\n"
                    )
            # Force browser flush with comment padding
            # Browser buffers until ~1KB received
            # This comment pushes past that threshold
            yield ": padding\n\n"
            # Send sources after answer completes
            # yield (
            #     "data: " +
            #     json.dumps({
            #         "sources":    list(set(sources)),
            #         "agent_used": "RAGPipeline",
            #         "done":       True,
            #     }) + "\n\n"
            # )

            # Signal stream complete
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield (
                "data: " +
                json.dumps({"error": str(e)}) +
                "\n\n"
            )
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """
    DELETE /documents/{doc_name}
    Remove all chunks for a specific document
    from Pinecone.
    """
    try:
        pc    = PineconeClient(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)

        # Pinecone delete by metadata filter
        index.delete(
            filter={"doc_name": {"$eq": doc_name}}
        )

        # Clear cache since docs changed
        get_cache().clear()

        return {
            "message": f"Deleted {doc_name}",
            "status":  "deleted"
        }
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.delete("/reset")
def reset_database():
    try:
        pc    = PineconeClient(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        index.delete(delete_all=True)
        return {"message": "Pinecone cleared", "status": "reset"}
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    print("BharatRAG API starting...")
    print(f"Model:     claude-sonnet-4-5")
    print(f"Vector DB: Pinecone ({PINECONE_INDEX})")
    chunks = get_pinecone_chunk_count()
    print(f"Chunks:    {chunks}")
    print("API ready at http://localhost:8000")
    print("Docs at http://localhost:8000/docs")

@app.get("/cache/stats")
def cache_stats():
    """GET /cache/stats — cache performance metrics"""
    return get_cache().stats()

@app.delete("/cache/clear")
def cache_clear():
    """DELETE /cache/clear — clear all cached answers"""
    get_cache().clear()
    return {"message": "Cache cleared"}