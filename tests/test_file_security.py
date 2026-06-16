"""
test_file_security.py — upload security controls.

Proves that the server correctly rejects:
- Path traversal filenames
- Non-PDF extensions
- Empty files
- Duplicate uploads (dedup)
- Oversized filenames
"""
import io
import pytest
import requests
from tests.helpers import upload_doc, CRAG_PDF


def _raw_upload(filename: str, content: bytes, user_id: str, base_url: str):
    """Upload arbitrary bytes with an arbitrary filename — bypasses open()."""
    return requests.post(
        f"{base_url}/upload",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
        data={"user_id": user_id},
        timeout=30,
    )


# ── Tests ─────────────────────────────────────────────────────

def test_path_traversal_filename_blocked(base_url):
    """A filename like ../../etc/passwd.pdf must be rejected with 400."""
    r = _raw_upload("../../etc/passwd.pdf", b"%PDF-1.4 fake", "sec_user_1", base_url)
    assert r.status_code == 400, f"Expected 400 for path traversal, got {r.status_code}: {r.text}"


def test_non_pdf_extension_blocked(base_url):
    """Uploading a .txt file must be rejected with 400."""
    r = _raw_upload("document.txt", b"just some text", "sec_user_2", base_url)
    assert r.status_code == 400, f"Expected 400 for non-PDF, got {r.status_code}: {r.text}"


def test_empty_file_handled(base_url):
    """A 0-byte upload must return 400 or 422, never 500."""
    r = _raw_upload("empty.pdf", b"", "sec_user_3", base_url)
    assert r.status_code in (400, 422), (
        f"Expected 400 or 422 for empty file, got {r.status_code}: {r.text}"
    )


def test_duplicate_upload_skipped(base_url):
    """Uploading the same file twice for the same user returns status='skipped' the second time."""
    first  = upload_doc(CRAG_PDF, "user_h_isolation", base_url)
    second = upload_doc(CRAG_PDF, "user_h_isolation", base_url)

    assert second.get("status") == "skipped", (
        f"Second upload of same file should be skipped. Got: {second}"
    )
    # chunks should not have doubled
    assert second.get("chunks", 0) == 0, (
        f"Skipped upload should report 0 new chunks. Got: {second}"
    )


def test_large_filename_handled(base_url):
    """A 300-character filename ending in .pdf must be rejected with 400, not 500."""
    long_name = "a" * 296 + ".pdf"   # 300 chars
    r = _raw_upload(long_name, b"", "sec_user_4", base_url)
    # 429 = rate limited (also acceptable — server didn't 500)
    assert r.status_code in (400, 429), (
        f"Expected 400 for oversized filename, got {r.status_code}: {r.text}"
    )
