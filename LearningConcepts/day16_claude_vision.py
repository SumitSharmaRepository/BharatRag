# ============================================
# DAY 16: Claude Vision for Scanned PDFs
# ============================================
# Days 1-15: Text PDFs only
# PyPDFLoader reads text layer
# Fails silently on scanned documents
#
# Day 16: Multimodal AI
# Convert PDF pages to images
# Send images to Claude Vision
# Claude reads visually — like a human
#
# Solves the biggest Indian market problem:
# Scanned government documents, court records,
# old ledger books, handwritten notes
# ============================================

import os
import base64
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import anthropic
from pdf2image import convert_from_path
from PIL import Image

# ── Setup ─────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
client        = anthropic.Anthropic(
    api_key=ANTHROPIC_KEY
)

# ============================================
# CORE FUNCTION: Image → Base64
# ============================================

def image_to_base64(image: Image.Image,
                    format: str = "JPEG") -> str:
    """
    Convert PIL Image to base64 string.

    Claude API requires images as base64 strings.
    Cannot send raw image bytes over JSON.

    Args:
        image:  PIL Image object
        format: JPEG or PNG

    Returns:
        base64 encoded string
    """
    import io
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.standard_b64encode(
        buffer.read()
    ).decode("utf-8")


# ============================================
# CORE FUNCTION: Claude Vision OCR
# ============================================

def extract_text_from_image(
    image:    Image.Image,
    language: str = "English"
) -> str:
    """
    Send image to Claude Vision and extract text.

    This is the key Day 16 function.
    Works on:
    → Scanned PDFs
    → Image-only PDFs
    → Handwritten documents
    → Hindi text
    → Tables and diagrams

    Args:
        image:    PIL Image of one PDF page
        language: Expected language in document

    Returns:
        Extracted text string
    """
    # Convert to base64
    image_b64 = image_to_base64(image)

    # Language-specific instruction
    lang_instruction = {
        "English":  "Extract all English text exactly.",
        "Hindi":    "इस दस्तावेज़ से सभी हिंदी "
                   "और अंग्रेजी टेक्स्ट निकालें।",
        "Auto":     "Extract all text. "
                   "Preserve the original language.",
    }.get(language, "Extract all text exactly.")

    # Send to Claude Vision
    response = client.messages.create(
        model      = "claude-sonnet-4-5",
        max_tokens = 4096,
        messages   = [{
            "role":    "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": "image/jpeg",
                        "data":       image_b64,
                    }
                },
                {
                    "type": "text",
                    "text": f"""Extract ALL text from this document image.

Instructions:
- {lang_instruction}
- Preserve formatting where possible
- Include all numbers, dates, amounts
- Include table contents row by row
- If text is unclear, make best attempt
- Output ONLY the extracted text
- No explanations or commentary"""
                }
            ]
        }]
    )

    return response.content[0].text


# ============================================
# PROCESS SCANNED PDF
# ============================================

def process_scanned_pdf(
    pdf_path: str,
    language: str = "Auto",
    dpi:      int = 200,
    pages:    list = None,
) -> list[dict]:
    """
    Convert scanned PDF to text using Claude Vision.

    Workflow:
    1. Convert each PDF page to image
    2. Send each image to Claude Vision
    3. Collect extracted text per page
    4. Return list of page results

    Args:
        pdf_path: Path to PDF file
        language: "English", "Hindi", or "Auto"
        dpi:      Image quality (higher = better but slower)
                  150 = fast, 200 = balanced, 300 = high quality
        pages:    List of page numbers to process
                  None = all pages

    Returns:
        List of dicts: {page, text, word_count}
    """
    pdf_name = Path(pdf_path).name
    print(f"\nProcessing: {pdf_name}")
    print(f"Language: {language}, DPI: {dpi}")

    # Convert PDF to images
    print("Converting PDF pages to images...")

    if pages:
        # Convert specific pages only
        images = convert_from_path(
            pdf_path,
            dpi         = dpi,
            first_page  = min(pages),
            last_page   = max(pages),
        )
        page_nums = pages
    else:
        # Convert all pages
        images    = convert_from_path(pdf_path, dpi=dpi)
        page_nums = list(range(1, len(images) + 1))

    print(f"Converted {len(images)} pages to images")

    # Extract text from each page
    results = []
    for i, (image, page_num) in enumerate(
        zip(images, page_nums)
    ):
        print(f"  Processing page {page_num}/{max(page_nums)}...",
              end=" ")

        text       = extract_text_from_image(image, language)
        word_count = len(text.split())

        print(f"{word_count} words extracted")

        results.append({
            "page":       page_num,
            "text":       text,
            "word_count": word_count,
            "source":     pdf_name,
        })

    total_words = sum(r["word_count"] for r in results)
    print(f"\nTotal: {total_words} words "
          f"from {len(results)} pages")

    return results


# ============================================
# COMPARE: PyPDFLoader vs Claude Vision
# ============================================

def compare_extraction_methods(pdf_path: str):
    """
    Compare text extraction quality:
    Method 1: PyPDFLoader (traditional)
    Method 2: Claude Vision (multimodal)

    Shows why Vision is better for scanned docs.
    """
    from langchain_community.document_loaders import PyPDFLoader

    print("\n" + "=" * 55)
    print("COMPARISON: PyPDFLoader vs Claude Vision")
    print("=" * 55)

    # Method 1: PyPDFLoader
    print("\nMethod 1: PyPDFLoader (text extraction)")
    loader   = PyPDFLoader(pdf_path)
    pages    = loader.load()
    py_text  = pages[0].page_content if pages else ""
    py_words = len(py_text.split())
    print(f"Words extracted: {py_words}")
    print(f"Preview: {py_text[:200]}...")

    # Method 2: Claude Vision (first page only for demo)
    print("\nMethod 2: Claude Vision (image understanding)")
    images      = convert_from_path(pdf_path, dpi=150,
                                    first_page=1, last_page=1)
    vision_text  = extract_text_from_image(
        images[0], language="Auto"
    )
    vision_words = len(vision_text.split())
    print(f"Words extracted: {vision_words}")
    print(f"Preview: {vision_text[:200]}...")

    # Comparison
    print("\n" + "-" * 55)
    print("RESULT:")
    if py_words < 10:
        print("⚠️  PyPDFLoader: near empty — scanned PDF")
        print("✅ Claude Vision: full text extracted")
        print("→ This PDF REQUIRES Vision API")
    elif vision_words > py_words * 1.2:
        print(f"✅ Claude Vision: {vision_words} words "
              f"(+{vision_words-py_words} more than PyPDF)")
        print("→ Vision extracts more content")
    else:
        print(f"PyPDFLoader: {py_words} words")
        print(f"Claude Vision: {vision_words} words")
        print("→ Similar quality — PDF has text layer")


# ============================================
# Q&A ON SCANNED DOCUMENT
# ============================================

def answer_from_scanned_pdf(
    pdf_path: str,
    question: str,
    language: str = "Auto",
    pages:    list = None,
) -> str:
    """
    Full pipeline: scanned PDF → Q&A

    This replaces the entire Day 4-9 pipeline
    for scanned documents that have no text layer.

    Steps:
    1. Convert PDF pages to images
    2. Extract text via Claude Vision
    3. Ask Claude the question with extracted text
    """
    # Step 1 & 2: Extract text from scanned PDF
    print(f"Extracting text from scanned PDF...")
    page_results = process_scanned_pdf(
        pdf_path, language, pages=pages
    )

    # Combine all extracted text
    combined_text = "\n\n".join([
        f"[Page {r['page']}]\n{r['text']}"
        for r in page_results
    ])

    if not combined_text.strip():
        return "Could not extract text from document."

    # Step 3: Answer the question
    print(f"\nAnswering: '{question}'")

    lang_instruction = {
        "English": "Answer in English.",
        "Hindi":   "हिंदी में जवाब दें।",
        "Hinglish": "Hinglish mein jawab do.",
        "Auto":    "Answer in the same language as the question.",
    }.get(language, "Answer clearly.")

    response = client.messages.create(
        model      = "claude-sonnet-4-5",
        max_tokens = 1024,
        messages   = [{
            "role":    "user",
            "content": f"""You are a document assistant.

Answer using ONLY the provided document text.
If not found say: "This information is not in the document."
{lang_instruction}
Always cite the page number.

Document text:
{combined_text}

Question: {question}

Answer:"""
        }]
    )

    return response.content[0].text


# ============================================
# DEMO: Process your SmartDocs PDF
# ============================================

def demo_with_existing_pdf():
    """
    Demo using SmartDocs PDF — even though it
    has a text layer, shows the Vision pipeline.
    Then compares both methods.
    """
    pdf_path = "/home/sumit/bharatrag/data/" \
               "SmartDocs_Complete_Learning_Guide.pdf"

    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        print("Please provide a PDF path")
        return

    print("=" * 55)
    print("Demo 1: Compare extraction methods")
    compare_extraction_methods(pdf_path)

    print("\n" + "=" * 55)
    print("Demo 2: Q&A on scanned PDF pipeline")
    print("Processing first 2 pages only for speed...")

    answer = answer_from_scanned_pdf(
        pdf_path  = pdf_path,
        question  = "What is SmartDocs AI?",
        language  = "English",
        pages     = [1, 2],
    )

    print(f"\nAnswer: {answer}")
    # NEW — smart hybrid detection per page
    print("\nDemo 3: Smart hybrid extraction")
    results = smart_extract(pdf_path, language="Auto")

    print("\nSample from smart extraction:")
    for r in results[:2]:
        print(f"Page {r['page']} via {r['method']}: "
              f"{r['text'][:100]}...")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 55)
    print("Day 16: Claude Vision for Scanned PDFs")
    print("=" * 55)

    # Run demo
    demo_with_existing_pdf()

    print("\n" + "=" * 55)
    print("Interactive mode")
    print("=" * 55)

    pdf_path = input(
        "\nEnter PDF path (or press Enter for default): "
    ).strip()

    if not pdf_path:
        pdf_path = "/home/sumit/bharatrag/data/" \
                   "SmartDocs_Complete_Learning_Guide.pdf"

    language = input(
        "Language (English/Hindi/Hinglish/Auto) "
        "[default: Auto]: "
    ).strip() or "Auto"

    pages_input = input(
        "Pages to process (e.g. 1,2,3) "
        "[default: first 3]: "
    ).strip()

    pages = [int(p) for p in pages_input.split(",") \
             if p.strip().isdigit()] or [1, 2, 3]

    print(f"\nProcessing pages: {pages}")
    results = process_scanned_pdf(
        pdf_path, language, pages=pages
    )

    print("\n" + "=" * 55)
    print("Q&A Mode — type 'exit' to quit")
    print("=" * 55)

    # Combine extracted text
    combined = "\n\n".join([
        f"[Page {r['page']}]\n{r['text']}"
        for r in results
    ])

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ["exit", "quit", ""]:
            break

        response = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 1024,
            messages   = [{
                "role":    "user",
                "content": f"""Answer from document only.
Cite page numbers. Be concise.

Document:
{combined}

Question: {question}
Answer:"""
            }]
        )
        print(f"\nAnswer: {response.content[0].text}")


def smart_extract(pdf_path: str,
                  language: str = "Auto") -> list[dict]:
    """
    Intelligently choose extraction method per page.
    Text layer detected → PyPDFLoader (fast, cheap)
    Scanned detected    → Claude Vision (accurate)

    This is production-grade PDF handling.
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    results      = []
    vision_count = 0
    text_count   = 0

    # Convert all pages to images for Vision fallback
    images = convert_from_path(pdf_path, dpi=150)

    for i, (page, image) in enumerate(
        zip(pages, images)
    ):
        word_count = len(page.page_content.split())

        if word_count >= 50:
            # Text layer exists — use PyPDFLoader
            text       = page.page_content
            method     = "pypdf"
            text_count += 1
        else:
            # Scanned or image-heavy — use Vision
            print(f"  Page {i+1}: scanned → using Vision")
            text         = extract_text_from_image(
                image, language
            )
            method       = "vision"
            vision_count += 1

        results.append({
            "page":   i + 1,
            "text":   text,
            "method": method,
            "source": Path(pdf_path).name,
        })

    print(f"\nExtraction summary:")
    print(f"  PyPDF pages:  {text_count}")
    print(f"  Vision pages: {vision_count}")

    return results


"""
process_scanned_pdf():
→ Call when you KNOW document is scanned
→ Example: government certificate upload
→ Skips PyPDFLoader entirely
→ Every page goes through Vision

smart_extract():
→ Call for ANY unknown PDF
→ Detects per page automatically
→ Best for production use
→ Saves API costs on text pages

extract_text_from_image():
→ Called internally by both above
→ You don't call this directly
→ Low-level Vision API wrapper




From Zero to Production-Ready AI Product
What You Will Learn
This guide walks throug...

Method 2: Claude Vision (image understanding)
Words extracted: 140
Preview: # SmartDocs AI → V3 | Every Line Explained

Built by Sumit Sharma · 2025

## What You Will Learn

This guide walks through every line of SmartDocs AI — an AI-powered PDF Q&A; tool built with Python, S...

-------------------------------------------------------
RESULT:
PyPDFLoader: 135 words
Claude Vision: 140 words
→ Similar quality — PDF has text layer

=======================================================
Demo 2: Q&A on scanned PDF pipeline
Processing first 2 pages only for speed...
Extracting text from scanned PDF...

Processing: SmartDocs_Complete_Learning_Guide.pdf
Language: English, DPI: 200
Converting PDF pages to images...
Converted 2 pages to images
  Processing page 1/2... 136 words extracted
  Processing page 2/2... 309 words extracted

Total: 445 words from 2 pages

Answering: 'What is SmartDocs AI?'

Answer: According to the document, SmartDocs AI is **an AI-powered PDF Q&A tool built with Python, Streamlit, and Claude API**.

**Page 1**
"""        