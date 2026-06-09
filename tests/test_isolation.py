"""
test_isolation.py — data isolation between users.

Proves that Pinecone per-user filtering works correctly:
- User B cannot read User A's documents
- Archive hides a doc from search
- Restore makes it searchable again
- Permanent delete removes it entirely
- /documents endpoint is scoped per user
"""
import pytest
import requests
from tests.helpers import upload_doc, query, delete_doc, CRAG_PDF


def _no_answer(text: str) -> bool:
    """True when the LLM couldn't find relevant content."""
    t = text.lower()
    return (
        "could not find" in t
        or "don't have" in t
        or "do not have" in t
        or "no information" in t
        or "no documents" in t
        or "no relevant" in t
        or "not find" in t
    )


# ── Tests ─────────────────────────────────────────────────────

def test_user_b_cannot_see_user_a_document(base_url):
    """User B querying after User A uploads should get no results."""
    upload_doc(CRAG_PDF, "user_a_isolation", base_url)

    resp = query("What is CRAG?", "user_b_isolation", base_url)
    answer  = resp.get("answer", "")
    sources = resp.get("sources", [])

    assert _no_answer(answer) or sources == [], (
        f"User B should not see User A's doc.\nAnswer: {answer}\nSources: {sources}"
    )


def test_user_can_see_own_document(base_url):
    """User C uploads and queries their own document — should get a real answer."""
    upload_doc(CRAG_PDF, "user_c_isolation", base_url)

    resp    = query("What is CRAG?", "user_c_isolation", base_url)
    answer  = resp.get("answer", "")
    sources = resp.get("sources", [])

    assert not _no_answer(answer), f"User C should find their own doc.\nAnswer: {answer}"
    assert sources, f"Expected non-empty sources for own document.\nAnswer: {answer}"


def test_archived_doc_not_searchable(base_url):
    """Archiving a document should make it invisible to queries."""
    upload_doc(CRAG_PDF, "user_d_isolation", base_url)
    delete_doc("CRAG.pdf", "user_d_isolation", "archive", base_url)

    resp   = query("What is CRAG?", "user_d_isolation", base_url)
    answer = resp.get("answer", "")

    assert _no_answer(answer), (
        f"Archived doc should not appear in search.\nAnswer: {answer}"
    )


def test_restored_doc_is_searchable(base_url):
    """After archiving then restoring a document it should be searchable again."""
    # Ensure clean state: upload (may return 'skipped' if already uploaded)
    upload_doc(CRAG_PDF, "user_d2_isolation", base_url)
    delete_doc("CRAG.pdf", "user_d2_isolation", "archive", base_url)

    # Confirm it's invisible
    before = query("What is CRAG?", "user_d2_isolation", base_url)
    assert _no_answer(before.get("answer", "")), "Doc should be invisible after archive"

    # Restore
    r = requests.post(
        f"{base_url}/documents/CRAG.pdf/restore",
        params={"user_id": "user_d2_isolation"},
        timeout=30,
    )
    assert r.status_code == 200

    # Should be searchable again
    after   = query("What is CRAG?", "user_d2_isolation", base_url)
    answer  = after.get("answer", "")
    sources = after.get("sources", [])

    assert not _no_answer(answer) or sources, (
        f"Restored doc should be searchable.\nAnswer: {answer}"
    )


def test_permanent_delete_removes_from_search(base_url):
    """Permanently deleted documents must not appear in any future query."""
    upload_doc(CRAG_PDF, "user_e_isolation", base_url)
    delete_doc("CRAG.pdf", "user_e_isolation", "permanent", base_url)

    resp   = query("What is CRAG?", "user_e_isolation", base_url)
    answer = resp.get("answer", "")

    assert _no_answer(answer), (
        f"Permanently deleted doc should be gone.\nAnswer: {answer}"
    )


def test_documents_endpoint_isolates_per_user(base_url):
    """GET /documents returns only docs belonging to the querying user."""
    upload_doc(CRAG_PDF, "user_f_isolation", base_url)

    # user_f should see CRAG.pdf
    r_f = requests.get(f"{base_url}/documents", params={"user_id": "user_f_isolation"}, timeout=30)
    assert r_f.status_code == 200
    active_f = r_f.json().get("active", [])
    assert "CRAG.pdf" in active_f, f"user_f should see CRAG.pdf. Got: {active_f}"

    # user_g (never uploaded anything) should see empty list
    r_g = requests.get(f"{base_url}/documents", params={"user_id": "user_g_isolation_never_uploaded"}, timeout=30)
    assert r_g.status_code == 200
    active_g = r_g.json().get("active", [])
    assert active_g == [], f"user_g should see no docs. Got: {active_g}"
