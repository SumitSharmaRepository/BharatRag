# ============================================
# src/loader.py — PDF loading and chunking
# Single responsibility: load PDFs, make chunks
# ============================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf(file_path: str) -> list:
    """
    Load a PDF file and return list of Document objects.
    Each Document = one page with page_content and metadata.

    Args:
        file_path: path to PDF file

    Returns:
        list of LangChain Document objects
    """
    print(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages  = loader.load()
    print(f"Loaded {len(pages)} pages")
    return pages


def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for retrieval.

    Why RecursiveCharacterTextSplitter?
    Tries to split on paragraphs first,
    then sentences, then words.
    Preserves meaning better than simple splitting.

    Args:
        documents: list of Document objects from loader

    Returns:
        list of smaller Document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks "
          f"(avg {sum(len(c.page_content) for c in chunks)//len(chunks)} chars)")
    return chunks


# src/loader.py

def load_and_chunk_pdf(file_path: str) -> list:
    """
    Smart loader — detects text vs scanned per page.
    Falls back to Vision for scanned pages.
    """
    from langchain.schema import Document

    # Try smart extraction
    results = smart_extract(file_path)

    # Convert to LangChain Document objects
    documents = []
    for r in results:
        doc = Document(
            page_content = r["text"],
            metadata     = {
                "source":   file_path,
                "page":     r["page"] - 1,
                "doc_name": Path(file_path).name,
                "method":   r["method"],
                # Track which pages needed Vision
            }
        )
        documents.append(doc)

    # Now chunk them
    return chunk_documents(documents)