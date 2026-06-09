import os
import hashlib
import tempfile
import sqlite3
import uuid as uuid_module
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


# ── Chat history — SQLite ──────────────────────────────────

CHAT_DB_PATH = "/tmp/bharatrag_chat.db"

def _init_chat_db():
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id        TEXT PRIMARY KEY,
            user_id   TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            sources   TEXT DEFAULT '[]',
            agent_used TEXT DEFAULT '',
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

class SaveMessageBody(BaseModel):
    user_id:    str
    role:       str
    content:    str
    sources:    list  = []
    agent_used: str   = ""

class ExportRequest(BaseModel):
    user_id:  str
    messages: list


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
        import asyncio
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        def _process_and_index():
            loader = PyPDFLoader(tmp_path)
            pages  = loader.load()

            for page in pages:
                page.metadata["doc_name"] = file.filename

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100
            )
            chunks = splitter.split_documents(pages)

            # ── PII detection + redaction ──────────
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

            # ── Stamp metadata ─────────────────────
            uploaded_at = datetime.now(timezone.utc).isoformat()
            for chunk in clean_chunks:
                chunk.metadata["user_id"]     = user_id
                chunk.metadata["file_hash"]   = file_hash
                chunk.metadata["uploaded_at"] = uploaded_at
                chunk.metadata["archived"]    = False

            # ── Embed + upsert to Pinecone ─────────
            vs = get_vectorstore()
            vs.add_documents(clean_chunks)

            return len(pages), len(clean_chunks)

        pages_count, chunk_count = await asyncio.to_thread(_process_and_index)

        return UploadResponse(
            filename=file.filename,
            pages=pages_count,
            chunks=chunk_count,
            total_chunks=chunk_count,
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

            unique_sources = list(set(sources))
            yield "data: " + json.dumps({
                "done":       True,
                "sources":    unique_sources,
                "agent_used": "RAGPipeline",
            }) + "\n\n"
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


# ── PDF generation ────────────────────────────────────────

def _escape_para(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
    )

def _generate_chat_pdf(messages: list, export_time: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_RIGHT, TA_CENTER
    from io import BytesIO

    buf = BytesIO()
    PAGE_W, _ = A4
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.5*cm, bottomMargin=2*cm,
    )
    W = PAGE_W - 4*cm

    C_DARK   = HexColor("#0f172a")
    C_PURPLE = HexColor("#6366f1")
    C_SLATE  = HexColor("#64748b")
    C_TEXT   = HexColor("#1e293b")
    C_BG     = HexColor("#f8fafc")
    C_MUTED  = HexColor("#94a3b8")

    S = {
        "title": ParagraphStyle("t",  fontName="Helvetica-Bold",    fontSize=20, textColor=white),
        "sub":   ParagraphStyle("s",  fontName="Helvetica",         fontSize=11, textColor=C_PURPLE),
        "date":  ParagraphStyle("d",  fontName="Helvetica",         fontSize=9,  textColor=C_MUTED, alignment=TA_RIGHT),
        "you":   ParagraphStyle("y",  fontName="Helvetica-Bold",    fontSize=8,  textColor=C_SLATE),
        "brag":  ParagraphStyle("br", fontName="Helvetica-Bold",    fontSize=8,  textColor=C_PURPLE),
        "msg":   ParagraphStyle("m",  fontName="Helvetica",         fontSize=10, textColor=C_TEXT,   leading=15),
        "src":   ParagraphStyle("sr", fontName="Helvetica-Oblique", fontSize=8,  textColor=C_SLATE),
        "foot":  ParagraphStyle("f",  fontName="Helvetica",         fontSize=8,  textColor=C_MUTED,  alignment=TA_CENTER),
        "ts":    ParagraphStyle("ts", fontName="Helvetica",         fontSize=8,  textColor=C_MUTED,  alignment=TA_RIGHT),
        "bullet": ParagraphStyle("bl", fontName="Helvetica",        fontSize=10, textColor=C_TEXT,   leading=15, leftIndent=14),
    }

    def _md(text: str) -> list:
        """Convert markdown to a list of ReportLab Paragraphs."""
        import re

        def escape(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def inline(s: str) -> str:
            s = escape(s)
            s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
            s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', s)
            return s

        result = []
        for block in re.split(r'\n{2,}', text.strip()):
            pending = []
            for line in block.split("\n"):
                if line.startswith("- "):
                    if pending:
                        result.append(Paragraph(inline(" ".join(pending)), S["msg"]))
                        pending = []
                    result.append(Paragraph(f"• {inline(line[2:].strip())}", S["bullet"]))
                else:
                    pending.append(line)
            if pending:
                result.append(Paragraph(inline(" ".join(pending)), S["msg"]))
        return result or [Paragraph("", S["msg"])]

    story = []

    # Header — single full-width column guarantees background fill
    hdr = Table(
        [[
            [
                Paragraph("BharatRAG", S["title"]),
                Paragraph("Chat Export", S["sub"]),
                Paragraph(export_time, S["date"]),
            ]
        ]],
        colWidths=[W],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_DARK),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
        ("TOPPADDING",    (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    story.append(hdr)
    story.append(HRFlowable(width="100%", thickness=3, color=C_PURPLE, spaceAfter=10))

    # Drop the welcome message (always first, always assistant)
    filtered = messages[1:] if messages and messages[0].get("role") == "assistant" else messages

    # Messages — flat tables with SPAN to avoid nested-width overflow
    for msg in filtered:
        role      = msg.get("role", "user")
        content   = msg.get("content", "")
        sources   = msg.get("sources", [])
        timestamp = msg.get("timestamp", "")

        if not content:
            continue

        ts_str = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                ts_str = dt.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                ts_str = timestamp

        content_paras = _md(content)

        if role == "user":
            # 2-col flat table: row 0 = [label | ts], row 1 = [content SPAN]
            rows = [
                [Paragraph("YOU", S["you"]), Paragraph(ts_str, S["ts"])],
                [content_paras, ""],
            ]
            box = Table(rows, colWidths=[W * 0.65, W * 0.35])
            box.setStyle(TableStyle([
                ("SPAN",          (0,1),  (-1,1)),
                ("BACKGROUND",    (0,0),  (-1,-1), C_BG),
                ("LEFTPADDING",   (0,0),  (-1,-1), 12),
                ("RIGHTPADDING",  (0,0),  (-1,-1), 12),
                ("TOPPADDING",    (0,0),  (-1,0),  8),
                ("BOTTOMPADDING", (0,0),  (-1,0),  4),
                ("TOPPADDING",    (0,1),  (-1,-1), 4),
                ("BOTTOMPADDING", (0,-1), (-1,-1), 10),
            ]))
            story.append(box)

        else:
            # 3-col flat table: col 0 = purple border strip (4 pt)
            # row 0 = [border | label | ts]
            # row 1 = [border | content SPAN]
            # row 2 = [border | sources SPAN]  (optional)
            BORD = 4
            CW   = [BORD, (W - BORD) * 0.65, (W - BORD) * 0.35]
            rows = [
                ["", Paragraph("BHARATRAG", S["brag"]), Paragraph(ts_str, S["ts"])],
                ["", content_paras, ""],
            ]
            span_cmds = [("SPAN", (1,1), (2,1))]
            if sources:
                src_text = ", ".join(str(s) for s in sources)
                rows.append(["", Paragraph(f"Source: {_escape_para(src_text)}", S["src"]), ""])
                span_cmds.append(("SPAN", (1,2), (2,2)))

            box = Table(rows, colWidths=CW)
            box.setStyle(TableStyle([
                # Border strip
                ("BACKGROUND",    (0,0),  (0,-1), C_PURPLE),
                ("LEFTPADDING",   (0,0),  (0,-1), 0),
                ("RIGHTPADDING",  (0,0),  (0,-1), 0),
                ("TOPPADDING",    (0,0),  (0,-1), 0),
                ("BOTTOMPADDING", (0,0),  (0,-1), 0),
                # Content area
                ("BACKGROUND",    (1,0),  (-1,-1), white),
                ("LEFTPADDING",   (1,0),  (-1,-1), 10),
                ("RIGHTPADDING",  (1,0),  (-1,-1), 10),
                ("TOPPADDING",    (1,0),  (-1,0),  8),
                ("BOTTOMPADDING", (1,0),  (-1,0),  4),
                ("TOPPADDING",    (1,1),  (-1,-1), 4),
                ("BOTTOMPADDING", (1,-1), (-1,-1), 8),
                ("VALIGN",        (0,0),  (-1,-1), "TOP"),
                *span_cmds,
            ]))
            story.append(box)

        story.append(Spacer(1, 8))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=C_SLATE, spaceBefore=6, spaceAfter=6))
    story.append(Paragraph("Generated by BharatRAG", S["foot"]))
    story.append(Paragraph("https://bharat-rag.vercel.app", S["foot"]))
    story.append(Paragraph(f"{len(filtered)} messages exported", S["foot"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Chat history endpoints ─────────────────────────────────

@app.get("/chat/history")
@limiter.limit("30/minute")
async def get_chat_history(request: Request, user_id: str, limit: int = 50):
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM chat_messages WHERE user_id = ? ORDER BY timestamp ASC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    msgs = [dict(r) for r in rows]
    for m in msgs:
        m["sources"] = json.loads(m["sources"])
    return msgs


@app.post("/chat/save")
@limiter.limit("30/minute")
async def save_chat_message(request: Request, body: SaveMessageBody):
    msg_id    = str(uuid_module.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.execute(
        "INSERT INTO chat_messages (id, user_id, role, content, sources, agent_used, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (msg_id, body.user_id, body.role, body.content, json.dumps(body.sources), body.agent_used, timestamp),
    )
    conn.commit()
    conn.close()
    return {"id": msg_id, "saved": True}


@app.delete("/chat/clear")
@limiter.limit("5/hour")
async def clear_chat_history(request: Request, user_id: str):
    conn    = sqlite3.connect(CHAT_DB_PATH)
    result  = conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
    count   = result.rowcount
    conn.commit()
    conn.close()
    return {"cleared": True, "count": count}


@app.post("/chat/export")
@limiter.limit("10/hour")
async def export_chat_pdf(request: Request, body: ExportRequest):
    try:
        export_time = datetime.now(timezone.utc).strftime("%d %b %Y, %I:%M %p UTC")
        date_str    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        pdf_bytes   = _generate_chat_pdf(body.messages, export_time)
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="BharatRAG_Chat_{date_str}.pdf"'},
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"PDF generation failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    _init_chat_db()
    print("BharatRAG API starting...", flush=True)
    print(f"Model:     claude-sonnet-4-5", flush=True)
    print(f"Vector DB: Pinecone ({PINECONE_INDEX})", flush=True)

    # Pre-warm both Pinecone connections so the first upload doesn't pay init cost
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: (
            _get_pinecone_index(),                  # index singleton
            get_embeddings().embed_query("warmup"), # embeddings + inference client
        ))
        print("Pinecone clients warmed up", flush=True)
    except Exception as e:
        print(f"Warmup skipped: {e}", flush=True)

    print("API ready", flush=True)
    print(f"Docs at /docs", flush=True)


@app.get("/cache/stats")
def cache_stats():
    return get_cache().stats()

@app.delete("/cache/clear")
def cache_clear():
    get_cache().clear()
    return {"message": "Cache cleared"}
