# ============================================
# whatsapp.py — BharatRAG WhatsApp Integration
# ============================================
# Flow:
# User sends WhatsApp message
# → Twilio calls POST /whatsapp
# → We run BharatRAG pipeline
# → We send answer back via Twilio
# → User sees answer on WhatsApp
#
# Same RAG pipeline as web UI.
# Same language support.
# Same document knowledge base.
# Just different input/output channel.
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from pinecone import Pinecone as PineconeClient

# ── Config ────────────────────────────────────────────
ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "bharatrag")
TWILIO_SID       = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM      = os.getenv(
    "TWILIO_WHATSAPP_FROM",
    "whatsapp:+14155238886"
)
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# ── App ───────────────────────────────────────────────
app = FastAPI(
    title       = "BharatRAG WhatsApp",
    description = "WhatsApp interface for BharatRAG",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ── Global instances ──────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

llm = ChatAnthropic(
    model             = "claude-sonnet-4-5",
    temperature       = 0,
    anthropic_api_key = ANTHROPIC_KEY,
)

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# ── Session memory ────────────────────────────────────
# Simple in-memory session per phone number
# Stores last 4 messages for context
# Resets when server restarts
# Day 17 Mem0 can replace this for persistence
sessions: dict = {}

# ── Language detection ────────────────────────────────
def detect_language(text: str) -> str:
    """
    Detect language from message.
    User can override by typing:
    /hindi, /english, /hinglish, /arabic
    """
    text_lower = text.lower().strip()

    # Explicit commands
    if text_lower.startswith("/hindi"):
        return "Hindi / हिंदी"
    if text_lower.startswith("/hinglish"):
        return "Hinglish"
    if text_lower.startswith("/arabic"):
        return "Arabic / عربي"
    if text_lower.startswith("/english"):
        return "English"

    # Auto-detect Hindi characters
    hindi_chars = sum(
        1 for c in text
        if '\u0900' <= c <= '\u097F'
    )
    if hindi_chars > 2:
        return "Hindi / हिंदी"

    # Auto-detect Arabic characters
    arabic_chars = sum(
        1 for c in text
        if '\u0600' <= c <= '\u06FF'
    )
    if arabic_chars > 2:
        return "Arabic / عربي"

    return "English"

# ── Help message ──────────────────────────────────────
HELP_MESSAGE = """🤖 *BharatRAG — AI Document Assistant*

Ask me anything about your uploaded documents.

*Language commands:*
/hindi — हिंदी में जवाब
/hinglish — Hinglish mein
/arabic — عربي
/english — English (default)

*Other commands:*
/help — Show this message
/clear — Clear conversation history

*Examples:*
- What is CRAG?
- /hindi session state kya hai?
- Summarise the document

Powered by Claude AI 🚀"""

# ── Core RAG function ─────────────────────────────────
def run_rag(
    question:    str,
    language:    str,
    phone:       str,
) -> str:
    """
    Run BharatRAG pipeline for WhatsApp message.
    Returns answer string for WhatsApp reply.
    """
    try:
        # Get vectorstore
        vectorstore = PineconeVectorStore(
            index_name = PINECONE_INDEX,
            embedding  = embeddings,
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        # Retrieve relevant chunks
        docs = retriever.invoke(question)

        if not docs:
            return (
                "❌ I could not find relevant "
                "information in the documents.\n\n"
                "Try uploading a PDF first via the "
                "web interface."
            )

        # Format context
        doc_texts = []
        sources   = []
        for doc in docs:
            page     = doc.metadata.get("page", "?")
            doc_name = doc.metadata.get(
                "doc_name", "unknown"
            )
            source   = f"{doc_name} (p.{int(page)+1})"
            sources.append(source)
            doc_texts.append(
                f"[{source}]\n{doc.page_content}"
            )

        context = "\n\n".join(doc_texts)

        # Get session history
        history = sessions.get(phone, [])
        history_text = ""
        if history:
            history_text = "\nPrevious messages:\n"
            for msg in history[-4:]:
                role    = msg["role"].upper()
                content = msg["content"][:100]
                history_text += f"{role}: {content}\n"

        # Language instruction
        lang_instruction = {
            "English":       "Answer in clear English.",
            "Hindi / हिंदी": "हिंदी में जवाब दें।",
            "Hinglish":      "Answer in Hinglish naturally.",
            "Arabic / عربي": "أجب باللغة العربية.",
        }.get(language, "Answer in clear English.")

        prompt = f"""You are BharatRAG — AI document
assistant for Indian professionals on WhatsApp.
{history_text}
{lang_instruction}

Keep answer concise for WhatsApp (under 300 words).
Use simple formatting — WhatsApp supports *bold* only.
Always cite document name.
If not found say: "Not found in documents."

Context:
{context}

Question: {question}

Answer:"""

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )
        answer = response.content

        # Update session memory
        if phone not in sessions:
            sessions[phone] = []

        sessions[phone].extend([
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ])

        # Keep only last 8 messages
        sessions[phone] = sessions[phone][-8:]

        # Add sources footer
        source_text = "\n\n📄 " + " | ".join(
            list(set(sources))[:2]
        )

        return answer + source_text

    except Exception as e:
        return f"❌ Error: {str(e)[:100]}"


# ── WhatsApp Webhook ──────────────────────────────────

@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
):
    """
    POST /whatsapp
    Twilio calls this when user sends WhatsApp message.

    From: user's WhatsApp number
          format: whatsapp:+919876543210
    Body: message text

    Returns: TwiML response (XML)
    Twilio reads this and sends the reply
    """
    phone    = From.replace("whatsapp:", "")
    message  = Body.strip()
    response = MessagingResponse()
    msg      = response.message()

    print(f"WhatsApp from {phone}: '{message}'")

    # Handle commands
    if message.lower() in ["/help", "help", "hi",
                            "hello", "namaste"]:
        msg.body(HELP_MESSAGE)
        return Response(
            content    = str(response),
            media_type = "application/xml"
        )

    if message.lower() == "/clear":
        sessions.pop(phone, None)
        msg.body("✅ Conversation history cleared!")
        return Response(
            content    = str(response),
            media_type = "application/xml"
        )

    # Detect language
    language = detect_language(message)

    # Clean command prefix from message
    for cmd in ["/hindi ", "/hinglish ",
                "/arabic ", "/english "]:
        if message.lower().startswith(cmd):
            message = message[len(cmd):].strip()
            break

    # Run RAG pipeline
    answer = run_rag(message, language, phone)

    msg.body(answer)

    print(f"Answer to {phone}: {answer[:100]}...")

    return Response(
        content    = str(response),
        media_type = "application/xml"
    )


@app.get("/whatsapp/health")
def whatsapp_health():
    return {
        "status":   "healthy",
        "sessions": len(sessions),
    }


# ── Run standalone ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "whatsapp:app",
        host    = "0.0.0.0",
        port    = 8001,
        reload  = True,
    )